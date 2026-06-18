# PicoCAST K7UAZ 2026-03-22 KEMX Candidate Report

## Summary

- Case ID: `k7uaz_20260322`
- Radar site: `KEMX`
- Scans analyzed: 36
- Scans inside telemetry window: 27
- Scans with possible returns: 36
- Candidate clusters scored: 254
- Candidate labels: {'moderate_candidate': 115, 'no_candidate': 78, 'weak_candidate': 48, 'strong_candidate': 13}

This report highlights near-track radar features for visual inspection. A high-priority candidate radar return is not a balloon association by itself and requires visual inspection and multi-radar confirmation.

## Top Candidates

| candidate_rank | scan_time_utc | search_window | candidate_score | candidate_label | horizontal_distance_km | vertical_distance_m | max_reflectivity_dbz | n_gates |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2026-03-22T20:28:40Z | normal | 0.734 | strong_candidate | 2.177 | 389.463 | -6.000 | 2 |
| 2 | 2026-03-22T22:13:30Z | loose | 0.715 | strong_candidate | 9.994 | 328.500 | 0.000 | 2 |
| 3 | 2026-03-22T20:00:31Z | tight | 0.715 | strong_candidate | 2.215 | 132.519 | -1.500 | 2 |
| 4 | 2026-03-22T21:59:43Z | loose | 0.707 | strong_candidate | 7.909 | 1027.067 | -1.000 | 2 |
| 5 | 2026-03-22T20:00:31Z | tight | 0.702 | strong_candidate | 2.055 | 169.519 | -4.500 | 2 |
| 6 | 2026-03-22T20:14:42Z | normal | 0.674 | strong_candidate | 5.578 | 24.833 | -6.500 | 2 |
| 7 | 2026-03-22T20:07:36Z | tight | 0.674 | strong_candidate | 1.219 | 117.844 | -6.500 | 5 |
| 8 | 2026-03-22T20:00:31Z | tight | 0.667 | strong_candidate | 2.689 | 185.981 | -5.000 | 2 |
| 9 | 2026-03-22T20:28:40Z | tight | 0.666 | strong_candidate | 2.177 | 389.463 | -6.000 | 2 |
| 10 | 2026-03-22T20:00:31Z | tight | 0.661 | strong_candidate | 0.869 | 1034.481 | -6.500 | 2 |

## Plot Paths

- Top candidate plots: `cases/k7uaz_20260322/outputs/candidates/KEMX/top_candidate_plots`
- Summary plots: `cases/k7uaz_20260322/outputs/candidates/KEMX/summary_plots`

## Interpretation Notes

- Use the plots to inspect whether each near-track radar feature is compact or part of a broader weather/clutter field.
- Prefer candidates that are close to the expected track, close in altitude, compact, and temporally plausible.
- Do not treat `strong_candidate` as conclusive evidence; it only means high-priority near-track radar candidate.
- Follow-up should compare neighboring radar sites and inspect Level II moments around the same scan times.

## Files

- Candidate scores: `cases/k7uaz_20260322/outputs/candidates/KEMX/candidate_scores.csv`
- Top candidates: `cases/k7uaz_20260322/outputs/candidates/KEMX/top_candidates.csv`
- Gate clusters: `cases/k7uaz_20260322/outputs/candidates/KEMX/gate_clusters.csv`
- Near-track gates: `cases/k7uaz_20260322/outputs/candidates/KEMX/near_track_gates.csv`
- Scan gate summary: `cases/k7uaz_20260322/outputs/candidates/KEMX/scan_gate_summary.csv`
