# PicoCAST Tonight Review Packet - k7uaz_20260322

## Summary

This packet ranks telemetry-consistent near-track radar features requiring visual inspection and multi-radar confirmation. It is a triage artifact for human review, not a detection claim.

## Counts

- Tracklet families: 5
- Cross-radar review candidates: 6
- Total review queue items: 11

## Top Review Queue

| Rank | Type | Item | Tracklets | Score | Reason | Plot |
| :---: | :--- | :--- | :--- | :---: | :--- | :--- |
| 1 | `cross_radar_association` | `A001` | `KEMX_T004;KIWA_T001` | 0.860 | strong_cross_radar_candidate with 8.8 min overlap, 9.8 km median horizontal separation, 144 m median altitude separation (16 raw association variants in this family pair) | plots/review_rank_01_A001.png |
| 2 | `cross_radar_association` | `A007` | `KEMX_T003;KIWA_T001` | 0.822 | moderate_cross_radar_candidate with 8.8 min overlap, 15.1 km median horizontal separation, 92 m median altitude separation (56 raw association variants in this family pair) | plots/review_rank_02_A007.png |
| 3 | `cross_radar_association` | `A012` | `KEMX_T004;KIWA_T007` | 0.820 | moderate_cross_radar_candidate with 23.1 min overlap, 15.9 km median horizontal separation, 87 m median altitude separation (4 raw association variants in this family pair) | plots/review_rank_03_A012.png |
| 4 | `cross_radar_association` | `A051` | `KEMX_T005;KIWA_T007` | 0.795 | moderate_cross_radar_candidate with 23.1 min overlap, 14.8 km median horizontal separation, 223 m median altitude separation (14 raw association variants in this family pair) | plots/review_rank_04_A051.png |
| 5 | `cross_radar_association` | `A078` | `KEMX_T006;KIWA_T001` | 0.780 | moderate_cross_radar_candidate with 8.8 min overlap, 16.9 km median horizontal separation, 233 m median altitude separation (8 raw association variants in this family pair) | plots/review_rank_05_A078.png |
| 6 | `cross_radar_association` | `A098` | `KEMX_T006;KIWA_T007` | 0.750 | moderate_cross_radar_candidate with 23.1 min overlap, 22.0 km median horizontal separation, 164 m median altitude separation (2 raw association variants in this family pair) | plots/review_rank_06_A098.png |
| 7 | `tracklet_family` | `KIWA_F001` | `KIWA_T004` | 0.798 | excellent_plausible_tracklet representative; 78 m median vertical mismatch, 9.4 km mean corridor distance | plots/review_rank_07_KIWA_F001.png |
| 8 | `tracklet_family` | `KEMX_F001` | `KEMX_T007` | 0.786 | excellent_plausible_tracklet representative; 39 m median vertical mismatch, 4.1 km mean corridor distance | plots/review_rank_08_KEMX_F001.png |
| 9 | `tracklet_family` | `KEMX_F002` | `KEMX_T010` | 0.714 | excellent_plausible_tracklet representative; 64 m median vertical mismatch, 5.8 km mean corridor distance | plots/review_rank_09_KEMX_F002.png |
| 10 | `tracklet_family` | `KEMX_F003` | `KEMX_T006` | 0.703 | excellent_plausible_tracklet representative; 88 m median vertical mismatch, 6.8 km mean corridor distance | plots/review_rank_10_KEMX_F003.png |

## Cross-Radar Candidates

| Rank | Association | Tracklets | Label | Overlap | Median H Sep | Median V Sep |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| 1 | `A001` | `KEMX_T004;KIWA_T001` | `strong_cross_radar_candidate` | 8.8 min | 9.8 km | 144 m |
| 2 | `A007` | `KEMX_T003;KIWA_T001` | `moderate_cross_radar_candidate` | 8.8 min | 15.1 km | 92 m |
| 3 | `A012` | `KEMX_T004;KIWA_T007` | `moderate_cross_radar_candidate` | 23.1 min | 15.9 km | 87 m |
| 4 | `A051` | `KEMX_T005;KIWA_T007` | `moderate_cross_radar_candidate` | 23.1 min | 14.8 km | 223 m |
| 5 | `A078` | `KEMX_T006;KIWA_T001` | `moderate_cross_radar_candidate` | 8.8 min | 16.9 km | 233 m |
| 6 | `A098` | `KEMX_T006;KIWA_T007` | `moderate_cross_radar_candidate` | 23.1 min | 22.0 km | 164 m |

## Interpretation Guardrails

- Treat these rows as candidate radar returns near a known telemetry corridor.
- Prioritize visual inspection of the ranked plots and dashboard before any modeling.
- Do not use this packet to identify the balloon by itself; radar artifacts, clutter, and weather-adjacent returns remain possible.
- The next scientific step is visual inspection and multi-radar confirmation.
