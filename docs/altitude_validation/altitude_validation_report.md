# Altitude Validation Report — k7uaz_20260322

**Radar site:** KEMX
**Generated:** 2026-06-18 00:12 UTC

---

## Position Uncertainty Context

The balloon horizontal position is estimated from 6-character Maidenhead grid
squares. At this latitude, one grid square is several kilometers across, so
horizontal offsets of a few km may still be consistent with the telemetry
region.

Balloon altitude, by contrast, is reported directly in the telemetry and is
treated as substantially more reliable than the horizontal position.
**Altitude consistency is therefore used as the primary discriminator for
candidate quality.**

Expected altitude at each radar scan time is **linearly interpolated** from
the telemetry points. This interpolation assumes roughly constant ascent rate
between telemetry reports, which may introduce small errors if the balloon
experienced altitude changes between reports.

> **Radar beam-center caveat:** The radar gate altitude reported here is the
> *beam-center* altitude for each gate. The physical radar beam has finite
> width (beam-width), and this width increases with range from the radar.
> At the ranges involved (~50–200 km from KEMX), the beam may span several
> hundred meters vertically. A candidate whose beam-center altitude is off by
> a few hundred meters could still have the balloon within the beam volume.

---

## Altitude Consistency Summary

| Category | Count | % of 254 candidates |
|----------|------:|----:|
| Excellent (≤ 250 m) | 33 | 13.0% |
| Strong (≤ 500 m) | 12 | 4.7% |
| Moderate (≤ 1000 m) | 15 | 5.9% |
| Total candidates scored | 254 | 100% |

---

## Top 10 Altitude-Prioritized Candidates

| Alt Rank | Orig Rank | Scan Time | Alt Match | Vert Mismatch | H-dist | Score | Label |
|:--------:|:---------:|-----------|-----------|:-------------:|:------:|:-----:|-------|
| 1 | 195 | 22:47:59 | excellent altitude match | -156 m | 3.5 km | 0.198 | no_candidate |
| 2 | 209 | 22:47:59 | excellent altitude match | -156 m | 3.5 km | 0.156 | no_candidate |
| 3 | 25 | 20:35:45 | excellent altitude match | +53 m | 9.3 km | 0.600 | moderate_candidate |
| 4 | 6 | 20:14:42 | excellent altitude match | +25 m | 5.6 km | 0.674 | strong_candidate |
| 5 | 11 | 20:21:35 | excellent altitude match | -124 m | 5.8 km | 0.661 | strong_candidate |
| 6 | 220 | 22:47:59 | excellent altitude match | -156 m | 3.5 km | 0.133 | no_candidate |
| 7 | 3 | 20:00:31 | excellent altitude match | -133 m | 2.2 km | 0.715 | strong_candidate |
| 8 | 5 | 20:00:31 | excellent altitude match | -170 m | 2.1 km | 0.702 | strong_candidate |
| 9 | 8 | 20:00:31 | excellent altitude match | +186 m | 2.7 km | 0.667 | strong_candidate |
| 10 | 51 | 20:14:42 | excellent altitude match | -187 m | 4.6 km | 0.563 | moderate_candidate |

---

## Focus: Original Ranks 1, 3, 6, 7, 9

| Orig Rank | Alt Rank | Scan Time | Signed Vert | Abs Vert | Alt Label | Cand Score |
|:---------:|:--------:|-----------|:-----------:|:--------:|-----------|:----------:|
| 1 | 38 | 20:28:40 | -389 m | 389 m | strong altitude match | 0.734 |
| 3 | 7 | 20:00:31 | -133 m | 133 m | excellent altitude match | 0.715 |
| 6 | 4 | 20:14:42 | +25 m | 25 m | excellent altitude match | 0.674 |
| 7 | 13 | 20:07:36 | -118 m | 118 m | excellent altitude match | 0.674 |
| 9 | 39 | 20:28:40 | -389 m | 389 m | strong altitude match | 0.666 |

---

## Altitude Trend Assessment

The majority of focused candidates (ranks 1, 3, 6, 7, 9) have altitude mismatches within 1000 m of the expected balloon altitude, suggesting **altitude consistency with the telemetry profile**. This is a necessary but not sufficient condition for a genuine balloon return.

---

## Terminology

This analysis uses the term **"altitude-consistent near-track candidate"** to
describe radar returns that:
1. Appear near the expected Maidenhead grid-square region
2. Have radar gate altitudes consistent with the balloon telemetry altitude

This does **not** constitute a detection claim. Many factors — including
weather returns, ground clutter, and biological targets — can produce radar
returns at similar altitudes. The altitude consistency merely raises the
prior probability that a given candidate is the balloon.

---

## Caveats

- Horizontal position derives from 6-character Maidenhead grid squares
  (~4.6 × 7.1 km at this latitude). Distance-from-grid-center is not
  distance-from-balloon.
- Radar gate altitude is the beam-center altitude. The beam has finite
  width, increasing with range from KEMX. A mismatch of a few hundred
  meters does not necessarily mean the balloon was outside the beam.
- Expected altitude at scan time is linearly interpolated from telemetry
  points, assuming roughly constant ascent rate between reports.
- Altitude consistency is necessary but not sufficient evidence for
  identifying a picoballoon radar return.
