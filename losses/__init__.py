"""Risk-sensitive training losses for RC-IRSTD."""

from losses.calibrator_risk import (
    CalibratorRiskLossOutput,
    CurveCalibratorRiskLossOutput,
    FULL_BACKGROUND,
    WEIGHTED_STRATIFIED_BACKGROUND,
    calibrator_risk_capability_contract,
    curve_query_risk_aligned_calibrator_loss,
    log10_budget_curve_smoothness,
    query_risk_aligned_calibrator_loss,
    risk_aligned_calibrator_loss,
    surrogate_query_detection_probability,
    surrogate_query_pixel_false_alarm_rate,
)
from losses.hard_target_loss import (
    hard_target_miss_loss,
    image_hard_target_miss_risks,
    object_top_fraction_logits,
    object_top_fraction_scores,
)
from losses.local_peak_cvar import (
    aggregate_image_risks_by_domain,
    domain_pixel_tail_risks,
    domain_tail_risks,
    image_background_pixel_tail_risks,
    image_tail_risks,
    local_background_peak_logits,
    local_background_peak_scores,
    top_fraction_mean,
)
from losses.smooth_worst_domain import smooth_max, smooth_worst_domain
from losses.sls import SLSIoULoss, location_loss
from losses.target_background_margin import (
    DomainTailSeparationOutput,
    background_local_peak_mask,
    bottom_fraction_mean,
    dilate_target_mask,
    domain_tail_separation_loss,
    domain_target_background_margin_risks,
    image_target_background_margin_risks,
)

__all__ = [
    "CalibratorRiskLossOutput",
    "CurveCalibratorRiskLossOutput",
    "FULL_BACKGROUND",
    "WEIGHTED_STRATIFIED_BACKGROUND",
    "DomainTailSeparationOutput",
    "SLSIoULoss",
    "aggregate_image_risks_by_domain",
    "background_local_peak_mask",
    "bottom_fraction_mean",
    "calibrator_risk_capability_contract",
    "curve_query_risk_aligned_calibrator_loss",
    "dilate_target_mask",
    "domain_pixel_tail_risks",
    "domain_tail_separation_loss",
    "domain_tail_risks",
    "domain_target_background_margin_risks",
    "hard_target_miss_loss",
    "image_background_pixel_tail_risks",
    "image_hard_target_miss_risks",
    "image_tail_risks",
    "image_target_background_margin_risks",
    "local_background_peak_logits",
    "local_background_peak_scores",
    "location_loss",
    "log10_budget_curve_smoothness",
    "object_top_fraction_logits",
    "object_top_fraction_scores",
    "query_risk_aligned_calibrator_loss",
    "risk_aligned_calibrator_loss",
    "smooth_max",
    "smooth_worst_domain",
    "surrogate_query_detection_probability",
    "surrogate_query_pixel_false_alarm_rate",
    "top_fraction_mean",
]
