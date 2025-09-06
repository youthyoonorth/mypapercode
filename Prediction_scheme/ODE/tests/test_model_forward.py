import torch
from Prediction_scheme.ODE.model_ODE import Model_3D


def _run_forward(beams, out_feature=None):
    model = Model_3D(out_feature=out_feature)
    x = torch.randn(2, 2, 101, beams)
    # use smaller timespans/pre_points for faster unit tests
    y = model(x, timespans=2, pre_points=3)
    assert y.shape == (3, 2, 2, beams)
    assert model.out_feature == beams


def test_forward_64_beams():
    _run_forward(64, out_feature=64)


def test_forward_512_beams():
    _run_forward(512, out_feature=512)


def test_forward_infer_from_input():
    _run_forward(64, out_feature=None)
