from coreml_suite.controlnet import no_control


def test_no_control():
    expected_inputs = {
        "additional_residual_0": {"shape": (2, 2, 2)},
        "additional_residual_1": {"shape": (2, 4, 4)},
        "additional_residual_2": {"shape": (2, 8, 8)},
    }

    residual_kwargs = no_control(expected_inputs)

    assert len(residual_kwargs) == 3
    assert residual_kwargs["additional_residual_0"].shape == (2, 2, 2)
    assert residual_kwargs["additional_residual_1"].shape == (2, 4, 4)
    assert residual_kwargs["additional_residual_2"].shape == (2, 8, 8)
