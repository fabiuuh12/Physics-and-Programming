"""Navigation confidence and risk scoring."""

from __future__ import annotations

import numpy as np


def covariance_sigmas(covariance: np.ndarray) -> tuple[float, float]:
    p_rr = covariance[:2, :2]
    p_vv = covariance[2:, 2:]
    sigma_r = float(np.sqrt(max(0.0, np.trace(p_rr))))
    sigma_v = float(np.sqrt(max(0.0, np.trace(p_vv))))
    return sigma_r, sigma_v


def navigation_risk(
    covariance: np.ndarray,
    geometry: float,
    missed: float,
    lighting: float,
    risk_config: dict,
) -> float:
    sigma_r, sigma_v = covariance_sigmas(covariance)
    eps = float(risk_config.get("epsilon", 1.0e-6))
    risk = (
        float(risk_config["w_r"]) * sigma_r / float(risk_config["r0"])
        + float(risk_config["w_v"]) * sigma_v / float(risk_config["v0"])
        + float(risk_config["w_g"]) / (geometry + eps)
        + float(risk_config["w_m"]) * missed
        + float(risk_config["w_l"]) * lighting
    )
    return float(max(0.0, risk))


def clipped_risk(risk: float, risk_config: dict) -> float:
    return float(np.clip(risk, 0.0, float(risk_config["risk_max"])))
