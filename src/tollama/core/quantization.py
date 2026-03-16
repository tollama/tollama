"""Model quantization and precision selection for tollama runners.

Provides utilities to quantize model weights after pull and to configure
runtime precision (fp16, bf16, int8) for reduced memory footprint and
faster inference on supported hardware.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from typing import Any, Literal

from .storage import TollamaPaths

logger = logging.getLogger(__name__)

PrecisionMode = Literal["fp32", "fp16", "bf16", "int8", "int4"]

# Families that support dynamic precision selection at load time.
_PRECISION_CAPABLE_FAMILIES: dict[str, set[PrecisionMode]] = {
    "torch": {"fp32", "fp16", "bf16"},
    "sundial": {"fp32", "fp16", "bf16"},
    "toto": {"fp32", "fp16", "bf16"},
    "timer": {"fp32", "fp16", "bf16"},
    "timemixer": {"fp32", "fp16", "bf16"},
    "lag_llama": {"fp32", "fp16", "bf16"},
    "timesfm": {"fp32", "fp16", "bf16"},
}

# Families that support post-hoc weight quantization.
_QUANTIZE_CAPABLE_FAMILIES: dict[str, set[PrecisionMode]] = {
    "torch": {"int8", "int4"},
    "sundial": {"int8"},
    "toto": {"int8"},
    "timer": {"int8"},
    "timemixer": {"int8"},
    "lag_llama": {"int8"},
}


@dataclass(frozen=True)
class QuantizationResult:
    """Result of a quantization operation."""

    model_name: str
    original_precision: str
    target_precision: PrecisionMode
    original_size_bytes: int
    quantized_size_bytes: int
    savings_pct: float
    output_path: str


@dataclass(frozen=True)
class PrecisionConfig:
    """Runtime precision configuration for a model."""

    mode: PrecisionMode
    compute_dtype: str
    storage_dtype: str
    requires_gpu: bool


def get_supported_precisions(family: str) -> list[PrecisionMode]:
    """Return precision modes supported by a model family."""
    runtime = set(_PRECISION_CAPABLE_FAMILIES.get(family, set()))
    quantize = set(_QUANTIZE_CAPABLE_FAMILIES.get(family, set()))
    return sorted(runtime | quantize)


def resolve_precision_config(mode: PrecisionMode) -> PrecisionConfig:
    """Map a precision mode to concrete dtype configuration."""
    configs: dict[PrecisionMode, PrecisionConfig] = {
        "fp32": PrecisionConfig(
            mode="fp32", compute_dtype="float32", storage_dtype="float32",
            requires_gpu=False,
        ),
        "fp16": PrecisionConfig(
            mode="fp16", compute_dtype="float16", storage_dtype="float16",
            requires_gpu=True,
        ),
        "bf16": PrecisionConfig(
            mode="bf16", compute_dtype="bfloat16", storage_dtype="bfloat16",
            requires_gpu=True,
        ),
        "int8": PrecisionConfig(
            mode="int8", compute_dtype="float16", storage_dtype="int8",
            requires_gpu=False,
        ),
        "int4": PrecisionConfig(
            mode="int4", compute_dtype="float16", storage_dtype="int4",
            requires_gpu=True,
        ),
    }
    if mode not in configs:
        raise ValueError(f"unsupported precision mode: {mode!r}")
    return configs[mode]


def quantize_model(
    model_name: str,
    target_precision: PrecisionMode,
    *,
    family: str | None = None,
    paths: TollamaPaths | None = None,
) -> QuantizationResult:
    """Quantize an installed model's weights to the target precision.

    This creates a quantized copy alongside the original weights.  The
    original checkpoint is preserved so the user can revert at any time.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for model quantization. "
            "Install with: pip install torch"
        ) from exc

    resolved_paths = paths or TollamaPaths.default()
    model_dir = resolved_paths.models_dir / model_name

    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    if family is not None:
        supported = set(_QUANTIZE_CAPABLE_FAMILIES.get(family, set()))
        if target_precision not in supported:
            raise ValueError(
                f"family {family!r} does not support {target_precision} quantization. "
                f"Supported: {sorted(supported)}"
            )

    # Find checkpoint files
    checkpoint_files = list(model_dir.glob("*.bin")) + list(model_dir.glob("*.safetensors"))
    if not checkpoint_files:
        checkpoint_files = list(model_dir.rglob("*.bin")) + list(model_dir.rglob("*.safetensors"))

    if not checkpoint_files:
        raise FileNotFoundError(f"no checkpoint files found in {model_dir}")

    original_size = sum(f.stat().st_size for f in checkpoint_files)
    quantized_dir = model_dir / f"quantized_{target_precision}"
    quantized_dir.mkdir(parents=True, exist_ok=True)

    quantized_size = 0
    for ckpt_file in checkpoint_files:
        relative = ckpt_file.relative_to(model_dir)
        output_path = quantized_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)

        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        quantized_state = _quantize_state_dict(state_dict, target_precision, torch)
        torch.save(quantized_state, output_path)
        quantized_size += output_path.stat().st_size

        logger.info("quantized %s → %s", ckpt_file.name, target_precision)

    # Write quantization metadata
    meta = {
        "model_name": model_name,
        "original_precision": "fp32",
        "target_precision": target_precision,
        "original_size_bytes": original_size,
        "quantized_size_bytes": quantized_size,
        "checkpoint_count": len(checkpoint_files),
    }
    meta_path = quantized_dir / "quantization_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    savings_pct = (1.0 - quantized_size / max(original_size, 1)) * 100.0

    return QuantizationResult(
        model_name=model_name,
        original_precision="fp32",
        target_precision=target_precision,
        original_size_bytes=original_size,
        quantized_size_bytes=quantized_size,
        savings_pct=round(savings_pct, 1),
        output_path=str(quantized_dir),
    )


def remove_quantized(
    model_name: str,
    target_precision: PrecisionMode,
    *,
    paths: TollamaPaths | None = None,
) -> bool:
    """Remove a quantized variant for a model.  Returns True if removed."""
    resolved_paths = paths or TollamaPaths.default()
    quantized_dir = resolved_paths.models_dir / model_name / f"quantized_{target_precision}"
    if not quantized_dir.is_dir():
        return False
    shutil.rmtree(quantized_dir)
    logger.info("removed quantized %s variant for %s", target_precision, model_name)
    return True


def list_quantized_variants(
    model_name: str,
    *,
    paths: TollamaPaths | None = None,
) -> list[dict[str, Any]]:
    """List all quantized variants for a model."""
    resolved_paths = paths or TollamaPaths.default()
    model_dir = resolved_paths.models_dir / model_name
    if not model_dir.is_dir():
        return []

    variants: list[dict[str, Any]] = []
    for child in sorted(model_dir.iterdir()):
        if child.is_dir() and child.name.startswith("quantized_"):
            meta_path = child / "quantization_meta.json"
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                variants.append(meta)
            else:
                variants.append({
                    "target_precision": child.name.replace("quantized_", ""),
                    "path": str(child),
                })
    return variants


def _quantize_state_dict(
    state_dict: dict[str, Any],
    target: PrecisionMode,
    torch_module: Any,
) -> dict[str, Any]:
    """Quantize a state dict to the target precision."""
    result: dict[str, Any] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch_module.Tensor):
            result[key] = tensor
            continue

        if tensor.dtype in (torch_module.bool, torch_module.int64, torch_module.int32):
            result[key] = tensor
            continue

        if target == "int8":
            scale = tensor.abs().max() / 127.0
            quantized = (tensor / scale.clamp(min=1e-8)).round().to(torch_module.int8)
            result[key] = quantized
            result[f"{key}.__scale__"] = scale
        elif target == "int4":
            scale = tensor.abs().max() / 7.0
            quantized = (tensor / scale.clamp(min=1e-8)).round().clamp(-8, 7).to(torch_module.int8)
            result[key] = quantized
            result[f"{key}.__scale__"] = scale
            result[f"{key}.__bits__"] = torch_module.tensor(4, dtype=torch_module.int8)
        else:
            result[key] = tensor

    return result
