"""Compatibility re-export; strict claim runtime lives in :mod:`losses.sls`."""

from losses.sls import SLSIoULoss, location_loss

__all__ = ["SLSIoULoss", "location_loss"]
