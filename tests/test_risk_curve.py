import torch

from rc_irstd.models.risk_curve import RiskCurvePredictor


def test_risk_curve_is_structurally_monotone():
    torch.manual_seed(0)
    model = RiskCurvePredictor(input_dim=12, num_thresholds=32, hidden_dim=16, dropout=0.0)
    output = model(torch.randn(7, 12))
    for curve in output.values():
        assert torch.all(curve[:, 1:] <= curve[:, :-1] + 1e-7)
