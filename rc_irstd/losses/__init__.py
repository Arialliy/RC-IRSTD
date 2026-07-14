from rc_irstd.losses.calibrator import (
    RiskAlignedLossOutput,
    risk_aligned_calibrator_loss,
    surrogate_query_pd,
    surrogate_query_pixel_risk,
)
from rc_irstd.losses.cvar import smooth_upper_max, smooth_worst_group, upper_cvar
from rc_irstd.losses.quantile import budget_focused_weight, crossing_loss, pinball_loss
from rc_irstd.losses.risk_aware import RiskAwareDetectorLoss
from rc_irstd.losses.sls import SLSIoULoss, location_loss
from rc_irstd.losses.target_background_margin import (
    DomainTailSeparationDetectorLoss,
    DomainTailSeparationOutput,
    background_local_peak_mask,
    domain_tail_separation_loss,
    risk_ramp_weight,
)

__all__ = [
    "DomainTailSeparationDetectorLoss",
    "DomainTailSeparationOutput",
    "RiskAlignedLossOutput",
    "RiskAwareDetectorLoss",
    "SLSIoULoss",
    "background_local_peak_mask",
    "budget_focused_weight",
    "crossing_loss",
    "domain_tail_separation_loss",
    "location_loss",
    "pinball_loss",
    "risk_aligned_calibrator_loss",
    "risk_ramp_weight",
    "smooth_upper_max",
    "smooth_worst_group",
    "surrogate_query_pd",
    "surrogate_query_pixel_risk",
    "upper_cvar",
]
