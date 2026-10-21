"""FlashDeBERTa models vs Hugging Face's eager DeBERTa-v2 with the same weights, in packed and padded modes."""
import pytest
import torch
from transformers import DebertaV2Config, DebertaV2Model

from flashdeberta.model import FlashDebertaV2ForSequenceClassification, FlashDebertaV2Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_config(conv=False, **kwargs):
    return DebertaV2Config(
        hidden_size=128, num_attention_heads=2, intermediate_size=256, num_hidden_layers=3,
        relative_attention=True, position_buckets=256, max_relative_positions=-1, pos_att_type=["p2c", "c2p"],
        share_att_key=True, norm_rel_ebd="layer_norm", max_position_embeddings=512, position_biased_input=False,
        vocab_size=500, hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
        conv_kernel_size=3 if conv else 0, **kwargs,
    )


def make_batch(L, lengths, side="right", seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(1, 500, (len(lengths), L), device="cuda", generator=g)
    positions = torch.arange(L, device="cuda")[None]
    n = torch.tensor(lengths, device="cuda")[:, None]
    mask = (positions < n) if side == "right" else (positions >= L - n)
    return ids, mask.long()


def rel_err(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def run(model, ids, mask):
    model.zero_grad()
    out = model(input_ids=ids, attention_mask=mask).last_hidden_state * mask[..., None]
    (out * torch.linspace(-1, 1, out.shape[-1], device="cuda")).sum().backward()
    return out.detach(), {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("unpad", [True, False])
@pytest.mark.parametrize("conv", [False, True])
@pytest.mark.parametrize("L,lengths,side", [
    (300, [300, 200, 37], "right"),
    (1100, [1100, 700, 1], "right"),
    (1100, [1100, 640, 77], "left"),
    (300, [300, 300], "right"),
])
def test_matches_huggingface(unpad, conv, L, lengths, side):
    if side == "left" and not unpad:
        pytest.skip("padded mode assumes right padding")
    torch.manual_seed(0)
    reference = DebertaV2Model(make_config(conv)).cuda().eval()
    flash = FlashDebertaV2Model(make_config(conv, unpad_inputs=unpad)).cuda().eval()
    flash.load_state_dict(reference.state_dict())
    ids, mask = make_batch(L, lengths, side)

    ref_out, ref_grads = run(reference, ids, mask)
    out, grads = run(flash, ids, mask)
    assert rel_err(out, ref_out) < 1e-3
    for name, g in ref_grads.items():
        if g.norm() > 0:
            assert rel_err(grads[name], g) < 1e-2, name


def test_modes_agree_on_outputs_and_hidden_states():
    torch.manual_seed(0)
    model = FlashDebertaV2Model(make_config()).cuda().eval()
    ids, mask = make_batch(200, [200, 150, 9])
    outputs = {}
    for unpad in (True, False):
        model.config.unpad_inputs = unpad
        with torch.no_grad():
            outputs[unpad] = model(input_ids=ids, attention_mask=mask, output_hidden_states=True, return_dict=False)
    (seq_p, hidden_p), (seq_d, hidden_d) = outputs[True], outputs[False]
    keep = mask[..., None]
    assert len(hidden_p) == len(hidden_d) == 4
    assert torch.allclose(seq_p * keep, seq_d * keep, atol=1e-4)
    for a, b in zip(hidden_p, hidden_d):
        assert a.shape == b.shape == (3, 200, 128)
        assert torch.allclose(a * keep, b * keep, atol=1e-4)
    assert torch.equal(seq_p, hidden_p[-1])


def test_sequence_classification_with_gradient_checkpointing():
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda()
    model.gradient_checkpointing_enable()
    model.train()
    ids, mask = make_batch(128, [128, 90, 31])
    logits = {}
    for unpad in (True, False):
        model.config.unpad_inputs = unpad
        model.zero_grad()
        out = model(input_ids=ids, attention_mask=mask, labels=torch.tensor([0, 1, 2], device="cuda"))
        out.loss.backward()
        logits[unpad] = out.logits.detach()
        assert all(p.grad is not None for p in model.deberta.encoder.layer.parameters())
    assert torch.allclose(logits[True], logits[False], atol=1e-4)


def count_syncs(fn):
    import warnings
    fn()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    return sum("synchroniz" in str(w.message) for w in caught)


@pytest.mark.parametrize("unpad,expected", [(True, 1), (False, 0)])
def test_host_syncs_per_forward(unpad, expected):
    model = FlashDebertaV2Model(make_config(unpad_inputs=unpad)).cuda()
    ids, mask = make_batch(300, [300, 120, 7])

    def step():
        model(input_ids=ids, attention_mask=mask).last_hidden_state.sum().backward()

    # Packed mode syncs once (to learn the number of real tokens); padded mode never does.
    assert count_syncs(step) == expected
