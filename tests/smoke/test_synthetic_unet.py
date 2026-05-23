"""Tier 1 smoke: convert a synthetic micro-UNet through coremltools and load
it back with python_coreml_stable_diffusion's CoreMLModel.

Purpose: catch API breakage in coremltools / ml-stable-diffusion *without*
needing a real SD checkpoint, the ANE, or a converted .mlmodelc on disk.
Runs in minutes on a hosted macOS-ARM runner (no Apple internal stuff).

What it asserts:
  - coremltools.convert still accepts the call shape we use today
  - the resulting .mlpackage round-trips through CoreMLModel
  - expected_inputs exposes the input names/shapes we declared
  - calling the model returns the named output (`noise_pred`)

Auto-skips on non-Apple-Silicon hosts so Tier 0 CI on Linux ignores it.
"""
import platform
import shutil

import numpy as np
import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason="Tier 1 requires macOS on Apple Silicon",
)


# Tiny shapes — large enough to exercise conv2d + linear + addition kernels in
# coremltools, small enough that conversion finishes in seconds on CPU.
SAMPLE_SHAPE = (1, 4, 8, 8)
TIMESTEP_SHAPE = (1,)
ENCODER_SHAPE = (1, 64, 1, 4)  # matches SD's transposed encoder_hidden_states layout
OUT_NAME = "noise_pred"


class TinyUNet(nn.Module):
    """Minimal UNet-shaped graph: conv -> add(time+context) -> conv.

    Not a real diffusion model. Just enough op variety to exercise the
    PyTorch -> MIL frontend in coremltools and confirm we can still wire
    the inputs/outputs the way ml-stable-diffusion expects.
    """

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(1, 8)
        self.text_proj = nn.Linear(64, 8)

    def forward(self, sample, timestep, encoder_hidden_states):
        h = self.conv_in(sample)
        t_emb = self.time_proj(timestep.unsqueeze(-1)).view(1, 8, 1, 1)
        c_emb = self.text_proj(encoder_hidden_states.squeeze(2).mean(-1)).view(1, 8, 1, 1)
        h = h + t_emb + c_emb
        return self.conv_out(h)


@pytest.fixture(scope="module")
def tiny_mlpackage(tmp_path_factory):
    """Convert TinyUNet once per test session and reuse the .mlpackage."""
    import coremltools as ct

    torch.manual_seed(0)
    model = TinyUNet().eval()
    example = (
        torch.randn(*SAMPLE_SHAPE),
        torch.randn(*TIMESTEP_SHAPE),
        torch.randn(*ENCODER_SHAPE),
    )
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="sample", shape=SAMPLE_SHAPE, dtype=np.float16),
            ct.TensorType(name="timestep", shape=TIMESTEP_SHAPE, dtype=np.float16),
            ct.TensorType(name="encoder_hidden_states", shape=ENCODER_SHAPE, dtype=np.float16),
        ],
        outputs=[ct.TensorType(name=OUT_NAME, dtype=np.float16)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
    )

    out_dir = tmp_path_factory.mktemp("tiny_unet")
    pkg_path = out_dir / "tiny.mlpackage"
    mlmodel.save(str(pkg_path))
    yield pkg_path
    shutil.rmtree(out_dir, ignore_errors=True)


def test_coremltools_convert_round_trips_via_coreml_model(tiny_mlpackage):
    from python_coreml_stable_diffusion.coreml_model import CoreMLModel

    model = CoreMLModel(str(tiny_mlpackage), "CPU_ONLY", "packages")

    # expected_inputs is the contract our wrappers depend on. Lock the shape
    # of the dict + a sample entry.
    expected = dict(model.expected_inputs)
    assert set(expected.keys()) == {"sample", "timestep", "encoder_hidden_states"}
    assert tuple(expected["sample"]["shape"]) == SAMPLE_SHAPE
    assert tuple(expected["timestep"]["shape"]) == TIMESTEP_SHAPE
    assert tuple(expected["encoder_hidden_states"]["shape"]) == ENCODER_SHAPE

    # Forward pass: drive the model the way CoreMLModelWrapper does.
    rng = np.random.default_rng(0)
    inputs = {
        "sample": rng.standard_normal(SAMPLE_SHAPE).astype(np.float16),
        "timestep": rng.standard_normal(TIMESTEP_SHAPE).astype(np.float16),
        "encoder_hidden_states": rng.standard_normal(ENCODER_SHAPE).astype(np.float16),
    }
    out = model(**inputs)
    assert isinstance(out, dict), f"unexpected output type: {type(out)}"
    assert OUT_NAME in out, f"missing output {OUT_NAME!r}; got {sorted(out)}"
    assert out[OUT_NAME].shape == SAMPLE_SHAPE, (
        f"output shape drift: got {out[OUT_NAME].shape}, expected {SAMPLE_SHAPE}"
    )
