"""FP8 linear layers (torchao) and FP8 attention, for inference and training, eager and compiled."""
import copy

import pytest
import torch
import torch._dynamo
from torch import nn
from torch._dynamo.utils import counters

from flashdeberta.model import FlashDebertaV2ForSequenceClassification, FlashDebertaV2Model
from flashdeberta.quantization import default_filter, prepare_fp8_training, quantize, set_attention_precision

from .test_model import make_batch, make_config

pytest.importorskip("torchao")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="requires a GPU with FP8 tensor cores",
)


@pytest.fixture(autouse=True)
def fresh_dynamo():
    torch._dynamo.reset()
    counters.clear()
    yield
    torch._dynamo.reset()


def masked_rel_err(a, b, mask):
    keep = mask[..., None]
    return (((a.float() - b.float()) * keep).norm() / (b.float() * keep).norm()).item()


def test_default_filter_keeps_heads_and_adapters_in_high_precision():
    linear = nn.Linear(128, 128)
    assert default_filter(linear, "deberta.encoder.layer.0.attention.self.query_proj")
    assert default_filter(linear, "deberta.encoder.layer.0.attention.self.query_proj.base_layer")
    assert not default_filter(linear, "classifier")
    assert not default_filter(linear, "pooler.dense")
    assert not default_filter(linear, "deberta.encoder.layer.0.attention.self.query_proj.lora_A.default")
    assert not default_filter(nn.Linear(100, 128), "deberta.encoder.layer.0.output.dense")


def test_set_attention_precision_validates():
    model = FlashDebertaV2Model(make_config())
    with pytest.raises(ValueError, match="precision must be one of"):
        set_attention_precision(model, "int8")
    with pytest.raises(ValueError, match="no FlashDeBERTa attention"):
        set_attention_precision(nn.Linear(2, 2), "fp8_qk")


@pytest.mark.parametrize("attention", [None, "fp8_qk", "fp8_qkv"])
def test_fp8_inference_compiled_with_cuda_graphs(attention):
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda().to(torch.bfloat16).eval()
    ids, mask = make_batch(300, [300, 120, 7])
    with torch.no_grad():
        expected = model.deberta(input_ids=ids, attention_mask=mask).last_hidden_state
    quantized = quantize(copy.deepcopy(model), linear="fp8_rowwise", attention=attention)
    assert isinstance(quantized.classifier.weight, nn.Parameter) and quantized.classifier.weight.dtype == torch.bfloat16
    assert type(quantized.deberta.encoder.layer[0].attention.self.query_proj.weight).__name__ != "Parameter"

    compiled = torch.compile(quantized, mode="reduce-overhead")
    with torch.no_grad():
        for _ in range(3):
            out = compiled(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            hidden = out.hidden_states[-1].clone()
    assert counters["inductor"]["cudagraph_skips"] == 0
    # Zeroed padding rows must not produce NaN through rowwise scaling.
    assert torch.isfinite(hidden).all() and torch.isfinite(out.logits).all()
    assert masked_rel_err(hidden, expected, mask) < 6e-2


@pytest.mark.parametrize("recipe", ["rowwise", "tensorwise"])
@pytest.mark.parametrize("attention", [None, "fp8_qk"])
def test_fp8_training_compiled(recipe, attention):
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda().to(torch.bfloat16).train()
    ids, mask = make_batch(300, [300, 120, 7])
    labels = torch.tensor([0, 1, 2], device="cuda")

    def step(m):
        model.zero_grad()
        out = m(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        return out.loss.item(), torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])

    loss, grads = step(model)
    prepare_fp8_training(model, recipe=recipe, attention=attention)
    assert sum(type(m).__name__ == "Float8Linear" for m in model.modules()) == 3 * 6  # 6 per layer
    compiled = torch.compile(model)
    for _ in range(2):
        fp8_loss, fp8_grads = step(compiled)
    assert abs(fp8_loss - loss) < 2e-2 * abs(loss)
    assert torch.isfinite(fp8_grads).all()
    assert torch.nn.functional.cosine_similarity(grads, fp8_grads, 0) > 0.99


def test_fp8_training_with_peft_lora():
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = FlashDebertaV2ForSequenceClassification(make_config(num_labels=3)).cuda().to(torch.bfloat16).train()
    prepare_fp8_training(model, recipe="rowwise", attention="fp8_qk")
    lora = peft.get_peft_model(model, peft.LoraConfig(r=8, target_modules=["query_proj", "value_proj"],
                                                      modules_to_save=["classifier", "pooler"]))
    ids, mask = make_batch(128, [128, 50])
    labels = torch.tensor([1, 2], device="cuda")
    compiled = torch.compile(lora)
    for _ in range(2):
        lora.zero_grad()
        loss = compiled(input_ids=ids, attention_mask=mask, labels=labels).loss
        loss.backward()
    trained = {n: p for n, p in lora.named_parameters() if p.requires_grad}
    assert any("lora_A" in n for n in trained) and all(p.grad is not None for p in trained.values())
    assert all(torch.isfinite(p.grad).all() for p in trained.values())
    # Frozen base layers run in FP8; the adapters stay in high precision.
    query = lora.base_model.model.deberta.encoder.layer[0].attention.self.query_proj
    assert type(query.base_layer).__name__ == "Float8Linear"
    assert isinstance(query.lora_A["default"], nn.Linear) and type(query.lora_A["default"]).__name__ == "Linear"
