# Candidate Trajectory Report — k7uaz_20260322

**Radar site:** KEMX
**Time window:** 2026-03-22T20:00:00Z → 2026-03-22T20:30:00Z
**Generated:** 2026-06-18 00:46 UTC

---

## Position Uncertainty Context

The balloon horizontal position is estimated from 6-character Maidenhead grid
squares. At this latitude, one grid square is several kilometers across, so
horizontal offsets of a few km may still be consistent with the telemetry
region. Altitude from telemetry is more reliable and is used as the primary
constraint.

Expected altitude at each radar scan time is **linearly interpolated** from
telemetry points.

> **Radar beam-center caveat:** Radar gate altitude is the beam-center
> altitude. The beam has finite width that increases with range. A candidate
> whose beam-center altitude differs by a few hundred meters could still
> have the balloon within the beam volume.

---

## Path Fit Summary

| Metric | Value |
|--------|-------|
| Candidate scan times in window | 5 |
| Selected path points | 5 |
| Time span | 20:00:31 → 20:28:40 UTC |
| Median absolute vertical mismatch | 118 m |
| Maximum segment speed | 72.7 km/h |
| All speeds plausible (< 250.0 km/h) | Yes ✓ |

---

## Selected Path Points

| Step | Scan Time | Orig R | Alt R | Signed Vert | H-dist | Speed | Alt Label |
|:----:|-----------|:------:|:-----:|:-----------:|:------:|:-----:|-----------|
| 1 | 20:00:31 | 3 | 7 | -133 m | 2.2 km | 0 km/h | excellent altitude match |
| 2 | 20:07:36 | 7 | 13 | -118 m | 1.2 km | 14 km/h | excellent altitude match |
| 3 | 20:14:42 | 6 | 4 | +25 m | 5.6 km | 73 km/h | excellent altitude match |
| 4 | 20:21:35 | 27 | 14 | -81 m | 5.9 km | 28 km/h | excellent altitude match |
| 5 | 20:28:40 | 15 | 37 | +465 m | 5.8 km | 21 km/h | strong altitude match |

    ---

    ## Assessment

    The selected candidates form an **altitude-consistent, horizontally smooth candidate sequence** that tracks the expected balloon altitude profile. Segment speeds are physically plausible for a picoballoon at these altitudes.

This constitutes an **altitude-constrained candidate trajectory** — a radar-assisted candidate path that is consistent with the balloon telemetry. It is not a confirmed detection, but the altitude consistency, horizontal smoothness, and plausible speeds raise the prior probability that these candidates are balloon-associated.

    ---

    ## Terminology

    This analysis uses cautious terminology:
    - **altitude-constrained candidate trajectory** — a path constructed from
      radar candidates whose altitudes are consistent with balloon telemetry
    - **radar-assisted candidate path** — the candidate path overlaid on radar
      geometry
    - **possible balloon-associated sequence** — candidates that may be related
      to the balloon, based on altitude and spatial consistency

    This does **not** use:
    - "confirmed track" — not confirmed without independent verification
    - "detected balloon" — altitude consistency is not proof of detection
    - "exact GPS track" — the horizontal position is from grid squares

    ---

    ## Caveats

    - Horizontal position from 6-character Maidenhead grid squares
      (~4.6 × 7.1 km at this latitude).
    - Radar gate altitude is beam-center altitude with beam-width uncertainty.
    - Expected altitude is linearly interpolated from sparse telemetry.
    - Path selection is greedy (not globally optimal).
    - Many non-balloon targets can appear at similar altitudes.
