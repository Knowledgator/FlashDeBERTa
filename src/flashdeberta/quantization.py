"""FP8 for FlashDeBERTa: linear layers through torchao, attention through the kernels' FP8 mode.

Inference::

    model = flashdeberta.quantization.quantize(model, linear="fp8_rowwise", attention="fp8_qk")
    model = torch.compile(model, mode="reduce-overhead")

Training::

    model = flashdeberta.quantization.prepare_fp8_training(model, recipe="tensorwise")
    model = torch.compile(model)

Attention precision is "native" (input dtype), "fp8_qk" (Q K^T in FP8) or "fp8_qkv" (P V too). It is
applied in the kernels, needs no extra memory and works with any backend. FP8 linear layers need
torchao and an FP8-capable GPU (compute capability 8.9+); they are meant to run under
``torch.compile``, which fuses the activation casts.

By default only the encoder's linear layers are converted. Heads (classifier, pooler, LM head),
LoRA adapter matrices (PEFT ``lora_A``/``lora_B``) and layers whose sizes are not multiples of 16
stay in high precision. Under PolyLoRA the base layer of every multi-LoRA linear is converted and
the adapters stay in the activation dtype.
"""
from __future__ import annotations

from typing import Callable, Optional

from torch import nn

from .model import FlashDisentangledSelfAttention
from .ops.fp8 import PRECISIONS

ModuleFilter = Callable[[nn.Module, str], bool]

_LORA_MARKERS = ("lora_A", "lora_B", "lora_embedding", "lora_magnitude")


def default_filter(module: nn.Module, fqn: str) -> bool:
    """Encoder linear layers with FP8-friendly sizes; no heads, no LoRA adapter matrices."""
    if not isinstance(module, nn.Linear):
        return False
    if "encoder" not in fqn.split("."):
        return False
    if any(marker in fqn for marker in _LORA_MARKERS):
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def set_attention_precision(model: nn.Module, precision: str) -> nn.Module:
    """Set the attention kernels' precision ("native", "fp8_qk", "fp8_qkv") on every FlashDeBERTa layer."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {precision!r}")
    found = False
    for module in model.modules():
        if isinstance(module, FlashDisentangledSelfAttention):
            module.attention_precision = precision
            found = True
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "attention_precision"):
            config.attention_precision = precision
    if not found:
        raise ValueError("the model has no FlashDeBERTa attention layers")
    return model


def quantize(model: nn.Module, linear: Optional[str] = "fp8_rowwise", attention: Optional[str] = None,
             module_filter: Optional[ModuleFilter] = None, tune_inductor: bool = False) -> nn.Module:
    """Quantize a model for FP8 inference, in place.

    Args:
        linear: "fp8_rowwise" (per-token activation and per-channel weight scales, the accurate
            choice), "fp8_tensorwise", or None to keep linear layers as they are.
        attention: attention precision, see ``set_attention_precision``; None leaves it unchanged.
        module_filter: ``(module, fqn) -> bool`` choosing the linear layers; default ``default_filter``.
        tune_inductor: let torchao switch on its global Inductor settings (coordinate-descent tuning of
            every Triton kernel). Off by default: it changes all later compiles in the process and makes
            them several times slower, while the FP8 GEMMs themselves run in cuBLAS.
    """
    if attention is not None:
        set_attention_precision(model, attention)
    if linear is None:
        return model
    from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerRow, PerTensor, quantize_

    granularity = {"fp8_rowwise": PerRow(), "fp8_tensorwise": PerTensor()}.get(linear)
    if granularity is None:
        raise ValueError(f"linear must be 'fp8_rowwise', 'fp8_tensorwise' or None, got {linear!r}")
    # DeBERTa zeroes padded tokens; a lower bound on the activation amax keeps their scale finite
    # (rowwise scaling would otherwise compute 0 / 0 and leak NaN into later layers).
    config = Float8DynamicActivationFloat8WeightConfig(granularity=granularity, activation_value_lb=1e-12,
                                                       set_inductor_config=tune_inductor)
    quantize_(model, config, filter_fn=module_filter or default_filter)
    return model


def prepare_fp8_training(model: nn.Module, recipe: str = "tensorwise", attention: Optional[str] = None,
                         module_filter: Optional[ModuleFilter] = None) -> nn.Module:
    """Convert linear layers to torchao FP8 training layers, in place.

    FP8 forward and backward GEMMs with dynamic scaling. ``recipe`` is a torchao recipe name:
    "tensorwise" (default; fastest in our measurements), "rowwise" or "rowwise_with_gw_hp" (more accurate
    scaling, but slower on GPUs without fast rowwise-scaled GEMMs such as SM120). With PEFT LoRA, convert the base model
    first and wrap it with PEFT afterwards: the frozen base layers run in FP8, the adapters stay in
    high precision. FP8 attention (``attention``) uses straight-through gradients.
    """
    if attention is not None:
        set_attention_precision(model, attention)
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    convert_to_float8_training(model, module_filter_fn=module_filter or default_filter,
                               config=Float8LinearConfig.from_recipe_name(recipe))
    return model
