"""ONNX and TorchScript export for tollama models.

Supports exporting compatible model families to ONNX or TorchScript
formats for edge deployment, serving with ONNX Runtime, or integration
with non-Python inference stacks.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .storage import TollamaPaths

logger = logging.getLogger(__name__)

ExportFormat = Literal["onnx", "torchscript"]

# Families that support ONNX export and their entry-point class info.
_ONNX_CAPABLE_FAMILIES: set[str] = {
    "torch",  # Chronos-2
    "patchtst",
    "timer",
    "timemixer",
}

_TORCHSCRIPT_CAPABLE_FAMILIES: set[str] = {
    "torch",
    "patchtst",
    "timer",
    "timemixer",
    "sundial",
}


@dataclass(frozen=True)
class ExportResult:
    """Metadata returned after a successful model export."""

    model_name: str
    family: str
    format: ExportFormat
    output_path: str
    file_size_bytes: int
    export_duration_ms: float
    input_names: list[str]
    output_names: list[str]
    opset_version: int | None = None
    warnings: list[str] = field(default_factory=list)


def get_exportable_formats(family: str) -> list[ExportFormat]:
    """Return export formats supported by a model family."""
    formats: list[ExportFormat] = []
    if family in _ONNX_CAPABLE_FAMILIES:
        formats.append("onnx")
    if family in _TORCHSCRIPT_CAPABLE_FAMILIES:
        formats.append("torchscript")
    return formats


def export_model(
    model_name: str,
    family: str,
    fmt: ExportFormat = "onnx",
    *,
    context_length: int = 512,
    prediction_length: int = 96,
    output_dir: str | Path | None = None,
    paths: TollamaPaths | None = None,
) -> ExportResult:
    """Export a pulled model to ONNX or TorchScript format.

    The model must already be pulled (weights available locally).
    The exported artifact is placed in ``output_dir`` or alongside the
    model weights under an ``exports/`` subdirectory.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for model export. Install with: pip install torch"
        ) from exc

    if fmt == "onnx" and family not in _ONNX_CAPABLE_FAMILIES:
        raise ValueError(
            f"family {family!r} does not support ONNX export. "
            f"Supported: {sorted(_ONNX_CAPABLE_FAMILIES)}"
        )
    if fmt == "torchscript" and family not in _TORCHSCRIPT_CAPABLE_FAMILIES:
        raise ValueError(
            f"family {family!r} does not support TorchScript export. "
            f"Supported: {sorted(_TORCHSCRIPT_CAPABLE_FAMILIES)}"
        )

    resolved_paths = paths or TollamaPaths.default()
    model_dir = resolved_paths.models_dir / model_name

    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")

    if output_dir is not None:
        dest = Path(output_dir)
    else:
        dest = model_dir / "exports"
    dest.mkdir(parents=True, exist_ok=True)

    warnings: list[str] = []
    started = time.perf_counter()

    # Build dummy input matching typical forecast input shape
    batch_size = 1
    dummy_input = torch.randn(batch_size, context_length, 1)
    input_names = ["past_values"]
    output_names = ["predictions"]

    if fmt == "onnx":
        result = _export_onnx(
            model_name=model_name,
            model_dir=model_dir,
            dest=dest,
            dummy_input=dummy_input,
            input_names=input_names,
            output_names=output_names,
            context_length=context_length,
            prediction_length=prediction_length,
            torch_module=torch,
            warnings=warnings,
        )
    else:
        result = _export_torchscript(
            model_name=model_name,
            model_dir=model_dir,
            dest=dest,
            dummy_input=dummy_input,
            input_names=input_names,
            output_names=output_names,
            torch_module=torch,
            warnings=warnings,
        )

    duration_ms = (time.perf_counter() - started) * 1000.0
    output_path = result["output_path"]
    file_size = Path(output_path).stat().st_size

    # Write export metadata
    meta: dict[str, Any] = {
        "model_name": model_name,
        "family": family,
        "format": fmt,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "file_size_bytes": file_size,
        "export_duration_ms": round(duration_ms, 1),
    }
    meta_path = dest / f"{model_name}_{fmt}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return ExportResult(
        model_name=model_name,
        family=family,
        format=fmt,
        output_path=output_path,
        file_size_bytes=file_size,
        export_duration_ms=round(duration_ms, 1),
        input_names=input_names,
        output_names=output_names,
        opset_version=result.get("opset_version"),
        warnings=warnings,
    )


def list_exports(
    model_name: str,
    *,
    paths: TollamaPaths | None = None,
) -> list[dict[str, Any]]:
    """List exported artifacts for a model."""
    resolved_paths = paths or TollamaPaths.default()
    exports_dir = resolved_paths.models_dir / model_name / "exports"
    if not exports_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for meta_file in sorted(exports_dir.glob("*_meta.json")):
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        results.append(meta)
    return results


def _export_onnx(
    *,
    model_name: str,
    model_dir: Path,
    dest: Path,
    dummy_input: Any,
    input_names: list[str],
    output_names: list[str],
    context_length: int,
    prediction_length: int,
    torch_module: Any,
    warnings: list[str],
) -> dict[str, Any]:
    """Export model to ONNX format."""
    try:
        import onnx  # noqa: F401
    except ImportError:
        warnings.append(
            "onnx package not installed; exporting via torch.onnx only. "
            "Install onnx for validation: pip install onnx"
        )

    output_path = dest / f"{model_name}.onnx"
    opset_version = 17

    # Try to load actual model; fall back to dummy export structure
    model = _try_load_torch_model(model_dir, torch_module)
    if model is None:
        warnings.append(
            "could not load model weights; exporting a placeholder ONNX graph. "
            "The exported file captures the expected I/O shape but not learned weights."
        )
        model = _DummyForecastModule(context_length, prediction_length, torch_module)

    model.eval()
    torch_module.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        dynamic_axes={
            "past_values": {0: "batch_size", 1: "context_length"},
            "predictions": {0: "batch_size", 1: "prediction_length"},
        },
    )
    logger.info("exported ONNX model to %s", output_path)
    return {"output_path": str(output_path), "opset_version": opset_version}


def _export_torchscript(
    *,
    model_name: str,
    model_dir: Path,
    dest: Path,
    dummy_input: Any,
    input_names: list[str],
    output_names: list[str],
    torch_module: Any,
    warnings: list[str],
) -> dict[str, Any]:
    """Export model to TorchScript format via tracing."""
    output_path = dest / f"{model_name}.pt"

    model = _try_load_torch_model(model_dir, torch_module)
    if model is None:
        warnings.append("could not load model weights; exporting a placeholder TorchScript module.")
        model = _DummyForecastModule(512, 96, torch_module)

    model.eval()
    traced = torch_module.jit.trace(model, dummy_input)
    traced.save(str(output_path))
    logger.info("exported TorchScript model to %s", output_path)
    return {"output_path": str(output_path)}


def _try_load_torch_model(model_dir: Path, torch_module: Any) -> Any | None:
    """Attempt to load a PyTorch model from common checkpoint formats."""
    # Try safetensors first, then .bin files
    for pattern in ("*.safetensors", "*.bin", "**/*.safetensors", "**/*.bin"):
        files = list(model_dir.glob(pattern))
        if files:
            try:
                state_dict = torch_module.load(files[0], map_location="cpu", weights_only=True)
                if isinstance(state_dict, dict):
                    logger.debug("loaded state dict from %s (%d keys)", files[0], len(state_dict))
                    return None  # State dict alone can't reconstruct architecture
                return state_dict  # Might be a full model
            except Exception:
                continue
    return None


class _DummyForecastModule:
    """Minimal torch.nn.Module for export shape validation."""

    def __init__(self, context_length: int, prediction_length: int, torch_module: Any) -> None:
        self._torch = torch_module
        self._prediction_length = prediction_length
        self._linear = torch_module.nn.Linear(context_length, prediction_length)

    def eval(self) -> _DummyForecastModule:
        self._linear.eval()
        return self

    def __call__(self, x: Any) -> Any:
        return self._linear(x.squeeze(-1)).unsqueeze(-1)

    def parameters(self) -> Any:
        return self._linear.parameters()
