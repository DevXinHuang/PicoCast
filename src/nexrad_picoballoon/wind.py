"""Wind context hooks for track scoring."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindVector:
    """Simple wind vector in meters per second."""

    u_ms: float
    v_ms: float


def constant_wind_prior(u_ms: float = 0.0, v_ms: float = 0.0) -> WindVector:
    """Return a simple placeholder wind prior for MVP tests and demos."""

    return WindVector(u_ms=u_ms, v_ms=v_ms)


def wind_consistency_score(observed_u_ms: float, observed_v_ms: float, prior: WindVector) -> float:
    """Score how closely observed motion follows a wind prior."""

    delta = ((observed_u_ms - prior.u_ms) ** 2 + (observed_v_ms - prior.v_ms) ** 2) ** 0.5
    return max(0.0, 1.0 - delta / 30.0)
