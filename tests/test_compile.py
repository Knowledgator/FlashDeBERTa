"""torch.compile: graph breaks, CUDA graphs, packed/padded/pre-packed layouts, training, FP8 attention."""
import pytest
import torch
import torch._dynamo
from torch._dynamo.utils import counters

from flashdeberta.model import FlashDebertaV2ForSequenceClassification, FlashDebertaV2Model
from flashdeberta.packing import PackedBatch, pack_inputs, packed_capacity
from flashdeberta.quantization import set_attention_precision

from .test_model import make_batch, make_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.config.error_on_recompile = False
    torch._dynamo.reset()


def masked_rel_err(a, b, mask):
    keep = mask[..., None]
    return (((a.float() - b.float()) * keep).norm() / (b.float() * keep).norm()).item()


def cudagraph_skips():
    return counters["inductor"]["cudagraph_skips"]


@pytest.mark.parametrize("unpad", [True, False])
def test_inference_is_one_graph_with_cuda_graphs(unpad):
    torch.manual_seed(0)
    model = FlashDebertaV2Model(make_config(unpad_inputs=unpad)).cuda().to(torch.bfloat16).eval()
    ids, mask = make_batch(300, [300, 120, 7])
    # Packing reads the token count before any compute, so the whole model is still one graph.
    explained = torch._dynamo.explain(model)(input_ids=ids, attention_mask=mask)
    assert explained.graph_count == 1

    torch._dynamo.reset()
    compiled = torch.compile(model, mode="reduce-overhead")
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
        for _ in range(3):  # warm-up, record, replay
            got = compiled(input_ids=ids, attention_mask=mask).last_hidden_state.clone()
    assert cudagraph_skips() == 0
    assert counters["inductor"]["cudagraph_recorded_non_static_inputs"] > 0
    assert masked_rel_err(got, expected, mask) < 2e-2


def test_padded_layout_compiles_with_fullgraph():
    model = FlashDebertaV2Model(make_config(unpad_inputs=False)).cuda().eval()
    ids, mask = make_batch(128, [128, 60])
    compiled = torch.compile(model, fullgraph=True)
    with torch.no_grad():
        got = compiled(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert masked_rel_err(got, expected, mask) < 1e-3


def test_packed_batches_of_different_sizes():
    """Every token count is rounded up to a bucket; outputs stay exact for all of them."""
    torch.manual_seed(0)
    model = FlashDebertaV2Model(make_config()).cuda().eval()
    compiled = torch.compile(model, mode="reduce-overhead")
    for lengths in ([200, 100, 3], [200, 150, 150], [200, 20, 1], [200, 199, 64]):
        ids, mask = make_batch(200, lengths, seed=sum(lengths))
        with torch.no_grad():
            expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for _ in range(2):
                got = compiled(input_ids=ids, attention_mask=mask).last_hidden_state.clone()
        assert masked_rel_err(got, expected, mask) < 1e-3, lengths
    assert cudagraph_skips() == 0


def test_prepacked_inputs_with_fullgraph_and_cuda_graphs():
    torch.manual_seed(0)
    model = FlashDebertaV2Model(make_config()).cuda().eval()
    ids, mask = make_batch(300, [300, 120, 7])
    packed = pack_inputs(ids, mask, device="cuda")
    assert packed["input_ids"].shape == (packed_capacity(427),)
    compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
    with torch.no_grad():
        expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
        for _ in range(3):
            got = compiled(**packed).last_hidden_state.clone()
    assert cudagraph_skips() == 0
    cu = packed["cu_seqlens"].tolist()
    for b in range(3):
        rows, ref = got[cu[b]:cu[b + 1]], expected[b, :cu[b + 1] - cu[b]]
        assert ((rows - ref).norm() / ref.norm()).item() < 1e-3


def test_prepacked_inputs_with_position_biased_input():
    """Packed position ids restart at every sequence."""
    torch.manual_seed(0)
    model = FlashDebertaV2Model(make_config()).cuda().eval()
    model.embeddings.position_biased_input = True  # make_config fixes it to False
    model.embeddings.position_embeddings = torch.nn.Embedding(512, 128).cuda()
    ids, mask = make_batch(64, [64, 30])
    packed = pack_inputs(ids, mask, bucket=None, device="cuda")
    with torch.no_grad():
        got = model(**packed).last_hidden_state
        expected = model(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(got[:64], expected[0], atol=1e-4)
    assert torch.allclose(got[64:], expected[1, :30], atol=1e-4)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_compiled_training_matches_eager(checkpointing):
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda().train()
    if checkpointing:
        model.gradient_checkpointing_enable()
    ids, mask = make_batch(300, [300, 120, 7])
    labels = torch.tensor([0, 1, 2], device="cuda")

    def step(m):
        model.zero_grad()
        out = m(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        return out.loss.detach(), {k: p.grad.clone() for k, p in model.named_parameters() if p.grad is not None}

    loss, grads = step(model)
    compiled = torch.compile(model)
    for _ in range(2):
        compiled_loss, compiled_grads = step(compiled)
    assert torch.allclose(compiled_loss, loss, rtol=1e-4)
    assert compiled_grads.keys() == grads.keys()
    for name, g in grads.items():
        if g.norm() > 0:
            assert ((compiled_grads[name] - g).norm() / g.norm()).item() < 1e-3, name


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 9),
                    reason="requires FP8 tensor cores")
def test_compiled_fp8_attention_training_matches_eager():
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda().to(torch.bfloat16).train()
    set_attention_precision(model, "fp8_qk")
    ids, mask = make_batch(200, [200, 77])
    labels = torch.tensor([0, 2], device="cuda")

    def step(m):
        model.zero_grad()
        out = m(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        return torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])

    eager = step(model)
    compiled = step(torch.compile(model))
    assert torch.nn.functional.cosine_similarity(eager, compiled, 0) > 0.999


def test_packed_capacity_buckets():
    assert packed_capacity(0) == 0
    assert packed_capacity(1) == 64 and packed_capacity(128) == 128 and packed_capacity(129) == 192
    assert packed_capacity(1025) == 1152  # an eighth of 1024 above 1024 tokens
    assert packed_capacity(5000, None) == 5000
    assert packed_capacity(100, 64) == 128
    assert packed_capacity(100, lambda n: 2 * n) == 200
    for n in range(1, 20000, 37):
        # At most 63 filler rows up to 1024 tokens, at most 12.5% beyond.
        assert 0 <= packed_capacity(n) - n < max(64, n / 8)
    with pytest.raises(ValueError):
        packed_capacity(100, lambda n: n - 1)


def test_packed_batch_round_trip_with_filler():
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], device="cuda").bool()
    batch = PackedBatch.from_mask(mask, capacity=6)
    x = torch.arange(8, device="cuda", dtype=torch.float32).view(2, 4, 1) + 1
    packed = batch.pack(x)
    assert packed.shape == (6, 1)
    assert packed[:4, 0].tolist() == [1, 2, 3, 5]
    assert batch.cu_seqlens.tolist() == [0, 3, 4]
    assert batch.token_mask().tolist() == [True] * 4 + [False] * 2
    assert batch.token_positions().tolist() == [0, 1, 2, 0, 0, 0]
    assert torch.equal(batch.unpack(packed), x * mask[..., None])
