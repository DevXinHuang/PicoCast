# PicoCAST Regional Discovery Report — Case k7uaz_20260322

## Executive Summary

This report summarizes the multi-radar balloon-like object discovery mode run. Instead of validation along a strict known-track line, we evaluated all visible regional NEXRAD sites for compact, altitude-plausible, weak point targets, connected them into candidate tracklets, and compared the results against balloon telemetry.

**Key Findings:**
- **Radars analyzed:** 2 sites included out of 5 total regional stations.
- **Discovered clusters:** 9371 compact radar returns within the corridor.
- **Linked tracklets:** 20 candidate tracklets linked across multiple scans.
- **Telemetry-consistent tracklets:** 20 candidate tracklets show close altitude-time agreement.
- **Cross-radar associations:** 100 associations where two radars see compatible trajectory behavior.

> [!NOTE]
> **Interpretation:** PicoCAST identified telemetry-consistent candidate tracklets with strong cross-radar candidate associations. This provides high prior probability of balloon-associated returns occurring in both KEMX and KIWA sweeps.

## Radars Evaluated & Geometry Status

| Radar Site | Location | Min Range (km) | Max Range (km) | Visible Scans | Total Scans | Geometry Status | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **KEMX** | Lat: 31.894, Lon: -110.630 | 48.1 | 87.0 | 27 | 36 | `include` | Included |
| **KIWA** | Lat: 33.289, Lon: -111.670 | 131.9 | 209.3 | 26 | 35 | `include` | Included |
| **KFSX** | Lat: 34.574, Lon: -111.198 | 251.8 | 288.9 | 0 | 29 | `skip` | Radar is too far (min distance 251.8 km > 250 km) |
| **KYUX** | Lat: 32.495, Lon: -114.657 | 352.5 | 456.6 | 0 | 29 | `skip` | Radar is too far (min distance 352.5 km > 250 km) |
| **KEPZ** | Lat: 31.873, Lon: -106.698 | 295.1 | 400.3 | 0 | 36 | `skip` | Radar is too far (min distance 295.1 km > 250 km) |

## NEXRAD Level II Ingest & Download Inventory

| Radar Site | Files Available | Files Downloaded | File Time Min | File Time Max | Total Size | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **KEMX** | 36 | 36 | 2026-03-22T18:49:37Z | 2026-03-22T22:54:53Z | 206.1 MB | `complete` |
| **KIWA** | 35 | 35 | 2026-03-22T18:51:02Z | 2026-03-22T22:53:07Z | 260.9 MB | `complete` |
| **KFSX** | 29 | 0 | 2026-03-22T18:49:18Z | 2026-03-22T22:55:46Z | 212.8 MB | `skipped` |
| **KYUX** | 29 | 0 | 2026-03-22T18:49:18Z | 2026-03-22T22:52:20Z | 282.1 MB | `skipped` |
| **KEPZ** | 36 | 0 | 2026-03-22T18:47:33Z | 2026-03-22T22:55:10Z | 223.8 MB | `skipped` |

## Cluster Extraction Statistics

We filtered raw gates to a piecewise linear expected-track corridor (40 km horizontal, ±1500 m vertical) and ran DBSCAN (eps=1.0 km, min_samples=1) to find compact candidates:

- **KEMX:** 6631 candidate clusters found
- **KIWA:** 2740 candidate clusters found

## Linked Candidate Tracklets

| Tracklet ID | Radar | Points | Start Time | End Time | Duration (min) | Med Vert Mismatch | Med Speed (km/h) | Label |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `KEMX_T001` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 73 m | 53.1 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T002` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 93 m | 42.0 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T003` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 29 m | 38.7 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T004` | KEMX | 7 | 19:53:25 | 20:49:55 | 56.5 | 46 m | 44.8 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T005` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 93 m | 42.0 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T006` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 88 m | 53.1 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T007` | KEMX | 8 | 19:53:25 | 20:57:01 | 63.6 | 39 m | 59.4 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T008` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 29 m | 38.7 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T009` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 113 m | 42.0 | `telemetry_consistent_candidate_tracklet` |
| `KEMX_T010` | KEMX | 6 | 19:53:25 | 20:28:40 | 35.2 | 64 m | 65.6 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T001` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 176 m | 29.9 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T002` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 176 m | 64.1 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T003` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 177 m | 39.8 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T004` | KIWA | 7 | 19:26:39 | 20:23:34 | 56.9 | 78 m | 44.8 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T005` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 176 m | 59.4 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T006` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 126 m | 39.9 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T007` | KIWA | 7 | 19:26:39 | 20:16:28 | 49.8 | 148 m | 43.3 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T008` | KIWA | 7 | 19:26:39 | 20:16:28 | 49.8 | 148 m | 53.1 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T009` | KIWA | 6 | 19:26:39 | 20:02:14 | 35.6 | 177 m | 64.1 | `telemetry_consistent_candidate_tracklet` |
| `KIWA_T010` | KIWA | 7 | 19:26:39 | 20:16:28 | 49.8 | 184 m | 42.1 | `telemetry_consistent_candidate_tracklet` |

## Cross-Radar Candidate Associations

Pairs of tracklets from different radars overlapping in time and sharing close horizontal/vertical trajectories:

| ID | Tracklets Associated | Overlap (min) | Med Horiz Diff | Med Alt Diff | Label |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `A001` | `KEMX_T004;KIWA_T001` | 8.8 | 9.83 km | 144 m | `strong_cross_radar_candidate` |
| `A002` | `KEMX_T004;KIWA_T002` | 8.8 | 9.83 km | 144 m | `strong_cross_radar_candidate` |
| `A003` | `KEMX_T004;KIWA_T003` | 8.8 | 9.76 km | 171 m | `strong_cross_radar_candidate` |
| `A004` | `KEMX_T004;KIWA_T009` | 8.8 | 9.76 km | 171 m | `strong_cross_radar_candidate` |
| `A005` | `KEMX_T004;KIWA_T005` | 8.8 | 10.21 km | 170 m | `moderate_cross_radar_candidate` |
| `A006` | `KEMX_T004;KIWA_T006` | 8.8 | 10.59 km | 224 m | `moderate_cross_radar_candidate` |
| `A007` | `KEMX_T003;KIWA_T001` | 8.8 | 15.05 km | 92 m | `moderate_cross_radar_candidate` |
| `A008` | `KEMX_T003;KIWA_T003` | 8.8 | 14.32 km | 119 m | `moderate_cross_radar_candidate` |
| `A009` | `KEMX_T008;KIWA_T001` | 8.8 | 15.05 km | 92 m | `moderate_cross_radar_candidate` |
| `A010` | `KEMX_T008;KIWA_T003` | 8.8 | 14.32 km | 119 m | `moderate_cross_radar_candidate` |
| `A011` | `KEMX_T005;KIWA_T001` | 8.8 | 12.67 km | 193 m | `moderate_cross_radar_candidate` |
| `A012` | `KEMX_T004;KIWA_T007` | 23.1 | 15.88 km | 87 m | `moderate_cross_radar_candidate` |
| `A013` | `KEMX_T003;KIWA_T002` | 8.8 | 15.05 km | 92 m | `moderate_cross_radar_candidate` |
| `A014` | `KEMX_T003;KIWA_T009` | 8.8 | 14.32 km | 119 m | `moderate_cross_radar_candidate` |
| `A015` | `KEMX_T008;KIWA_T002` | 8.8 | 15.05 km | 92 m | `moderate_cross_radar_candidate` |
| `A016` | `KEMX_T005;KIWA_T002` | 8.8 | 12.67 km | 193 m | `moderate_cross_radar_candidate` |
| `A017` | `KEMX_T008;KIWA_T009` | 8.8 | 14.32 km | 119 m | `moderate_cross_radar_candidate` |
| `A018` | `KEMX_T004;KIWA_T010` | 23.1 | 15.88 km | 87 m | `moderate_cross_radar_candidate` |
| `A019` | `KEMX_T004;KIWA_T008` | 23.1 | 15.88 km | 87 m | `moderate_cross_radar_candidate` |
| `A020` | `KEMX_T010;KIWA_T003` | 8.8 | 14.02 km | 119 m | `moderate_cross_radar_candidate` |
| `A021` | `KEMX_T010;KIWA_T001` | 8.8 | 15.00 km | 92 m | `moderate_cross_radar_candidate` |
| `A022` | `KEMX_T003;KIWA_T005` | 8.8 | 14.86 km | 118 m | `moderate_cross_radar_candidate` |
| `A023` | `KEMX_T005;KIWA_T003` | 8.8 | 12.65 km | 220 m | `moderate_cross_radar_candidate` |
| `A024` | `KEMX_T010;KIWA_T009` | 8.8 | 14.02 km | 119 m | `moderate_cross_radar_candidate` |
| `A025` | `KEMX_T008;KIWA_T005` | 8.8 | 14.86 km | 118 m | `moderate_cross_radar_candidate` |
| `A026` | `KEMX_T010;KIWA_T002` | 8.8 | 15.00 km | 92 m | `moderate_cross_radar_candidate` |
| `A027` | `KEMX_T005;KIWA_T009` | 8.8 | 12.65 km | 220 m | `moderate_cross_radar_candidate` |
| `A028` | `KEMX_T010;KIWA_T004` | 30.1 | 14.74 km | 86 m | `moderate_cross_radar_candidate` |
| `A029` | `KEMX_T010;KIWA_T005` | 8.8 | 14.55 km | 118 m | `moderate_cross_radar_candidate` |
| `A030` | `KEMX_T005;KIWA_T005` | 8.8 | 13.14 km | 219 m | `moderate_cross_radar_candidate` |
| `A031` | `KEMX_T007;KIWA_T003` | 8.8 | 17.05 km | 113 m | `moderate_cross_radar_candidate` |
| `A032` | `KEMX_T003;KIWA_T006` | 8.8 | 14.86 km | 175 m | `moderate_cross_radar_candidate` |
| `A033` | `KEMX_T004;KIWA_T004` | 30.1 | 17.57 km | 67 m | `moderate_cross_radar_candidate` |
| `A034` | `KEMX_T008;KIWA_T006` | 8.8 | 14.86 km | 175 m | `moderate_cross_radar_candidate` |
| `A035` | `KEMX_T009;KIWA_T001` | 8.8 | 14.98 km | 182 m | `moderate_cross_radar_candidate` |
| `A036` | `KEMX_T007;KIWA_T009` | 8.8 | 17.05 km | 113 m | `moderate_cross_radar_candidate` |
| `A037` | `KEMX_T007;KIWA_T001` | 8.8 | 16.63 km | 148 m | `moderate_cross_radar_candidate` |
| `A038` | `KEMX_T010;KIWA_T006` | 8.8 | 14.55 km | 175 m | `moderate_cross_radar_candidate` |
| `A039` | `KEMX_T009;KIWA_T002` | 8.8 | 14.98 km | 182 m | `moderate_cross_radar_candidate` |
| `A040` | `KEMX_T001;KIWA_T003` | 8.8 | 17.61 km | 99 m | `moderate_cross_radar_candidate` |
| `A041` | `KEMX_T007;KIWA_T002` | 8.8 | 16.63 km | 148 m | `moderate_cross_radar_candidate` |
| `A042` | `KEMX_T007;KIWA_T005` | 8.8 | 17.51 km | 113 m | `moderate_cross_radar_candidate` |
| `A043` | `KEMX_T001;KIWA_T001` | 8.8 | 17.01 km | 134 m | `moderate_cross_radar_candidate` |
| `A044` | `KEMX_T002;KIWA_T001` | 8.8 | 15.43 km | 195 m | `moderate_cross_radar_candidate` |
| `A045` | `KEMX_T005;KIWA_T004` | 30.1 | 16.00 km | 135 m | `moderate_cross_radar_candidate` |
| `A046` | `KEMX_T001;KIWA_T009` | 8.8 | 17.61 km | 99 m | `moderate_cross_radar_candidate` |
| `A047` | `KEMX_T003;KIWA_T004` | 30.1 | 15.54 km | 161 m | `moderate_cross_radar_candidate` |
| `A048` | `KEMX_T001;KIWA_T002` | 8.8 | 17.01 km | 134 m | `moderate_cross_radar_candidate` |
| `A049` | `KEMX_T005;KIWA_T006` | 8.8 | 13.56 km | 276 m | `moderate_cross_radar_candidate` |
| `A050` | `KEMX_T002;KIWA_T002` | 8.8 | 15.43 km | 195 m | `moderate_cross_radar_candidate` |
| `A051` | `KEMX_T005;KIWA_T007` | 23.1 | 14.78 km | 223 m | `moderate_cross_radar_candidate` |
| `A052` | `KEMX_T001;KIWA_T005` | 8.8 | 18.08 km | 99 m | `moderate_cross_radar_candidate` |
| `A053` | `KEMX_T008;KIWA_T004` | 30.1 | 15.77 km | 161 m | `moderate_cross_radar_candidate` |
| `A054` | `KEMX_T003;KIWA_T008` | 23.1 | 16.44 km | 152 m | `moderate_cross_radar_candidate` |
| `A055` | `KEMX_T005;KIWA_T010` | 23.1 | 14.78 km | 223 m | `moderate_cross_radar_candidate` |
| `A056` | `KEMX_T005;KIWA_T008` | 23.1 | 14.78 km | 223 m | `moderate_cross_radar_candidate` |
| `A057` | `KEMX_T008;KIWA_T008` | 23.1 | 16.44 km | 152 m | `moderate_cross_radar_candidate` |
| `A058` | `KEMX_T009;KIWA_T008` | 23.1 | 15.24 km | 212 m | `moderate_cross_radar_candidate` |
| `A059` | `KEMX_T009;KIWA_T003` | 8.8 | 15.26 km | 242 m | `moderate_cross_radar_candidate` |
| `A060` | `KEMX_T007;KIWA_T006` | 8.8 | 17.92 km | 160 m | `moderate_cross_radar_candidate` |
| `A061` | `KEMX_T003;KIWA_T010` | 23.1 | 17.28 km | 150 m | `moderate_cross_radar_candidate` |
| `A062` | `KEMX_T010;KIWA_T008` | 23.1 | 16.44 km | 151 m | `moderate_cross_radar_candidate` |
| `A063` | `KEMX_T008;KIWA_T010` | 23.1 | 17.28 km | 150 m | `moderate_cross_radar_candidate` |
| `A064` | `KEMX_T009;KIWA_T009` | 8.8 | 15.26 km | 242 m | `moderate_cross_radar_candidate` |
| `A065` | `KEMX_T009;KIWA_T007` | 23.1 | 16.06 km | 212 m | `moderate_cross_radar_candidate` |
| `A066` | `KEMX_T007;KIWA_T007` | 23.1 | 18.58 km | 145 m | `moderate_cross_radar_candidate` |
| `A067` | `KEMX_T009;KIWA_T010` | 23.1 | 16.07 km | 212 m | `moderate_cross_radar_candidate` |
| `A068` | `KEMX_T002;KIWA_T003` | 8.8 | 15.87 km | 244 m | `moderate_cross_radar_candidate` |
| `A069` | `KEMX_T007;KIWA_T010` | 23.1 | 18.58 km | 145 m | `moderate_cross_radar_candidate` |
| `A070` | `KEMX_T001;KIWA_T006` | 8.8 | 18.50 km | 151 m | `moderate_cross_radar_candidate` |
| `A071` | `KEMX_T009;KIWA_T005` | 8.8 | 15.75 km | 241 m | `moderate_cross_radar_candidate` |
| `A072` | `KEMX_T007;KIWA_T008` | 23.1 | 18.58 km | 145 m | `moderate_cross_radar_candidate` |
| `A073` | `KEMX_T010;KIWA_T007` | 23.1 | 17.27 km | 151 m | `moderate_cross_radar_candidate` |
| `A074` | `KEMX_T002;KIWA_T009` | 8.8 | 15.87 km | 244 m | `moderate_cross_radar_candidate` |
| `A075` | `KEMX_T009;KIWA_T006` | 8.8 | 16.07 km | 246 m | `moderate_cross_radar_candidate` |
| `A076` | `KEMX_T002;KIWA_T004` | 30.1 | 16.93 km | 185 m | `moderate_cross_radar_candidate` |
| `A077` | `KEMX_T010;KIWA_T010` | 23.1 | 17.28 km | 150 m | `moderate_cross_radar_candidate` |
| `A078` | `KEMX_T006;KIWA_T001` | 8.8 | 16.94 km | 233 m | `moderate_cross_radar_candidate` |
| `A079` | `KEMX_T002;KIWA_T008` | 23.1 | 16.44 km | 223 m | `moderate_cross_radar_candidate` |
| `A080` | `KEMX_T002;KIWA_T005` | 8.8 | 16.33 km | 244 m | `moderate_cross_radar_candidate` |
| `A081` | `KEMX_T006;KIWA_T002` | 8.8 | 16.94 km | 233 m | `moderate_cross_radar_candidate` |
| `A082` | `KEMX_T001;KIWA_T010` | 23.1 | 18.89 km | 150 m | `moderate_cross_radar_candidate` |
| `A083` | `KEMX_T008;KIWA_T007` | 23.1 | 17.27 km | 212 m | `moderate_cross_radar_candidate` |
| `A084` | `KEMX_T001;KIWA_T008` | 23.1 | 18.89 km | 152 m | `moderate_cross_radar_candidate` |
| `A085` | `KEMX_T009;KIWA_T004` | 30.1 | 16.93 km | 214 m | `moderate_cross_radar_candidate` |
| `A086` | `KEMX_T003;KIWA_T007` | 23.1 | 17.27 km | 223 m | `moderate_cross_radar_candidate` |
| `A087` | `KEMX_T002;KIWA_T006` | 8.8 | 16.33 km | 278 m | `moderate_cross_radar_candidate` |
| `A088` | `KEMX_T002;KIWA_T010` | 23.1 | 17.28 km | 223 m | `moderate_cross_radar_candidate` |
| `A089` | `KEMX_T002;KIWA_T007` | 23.1 | 17.27 km | 240 m | `moderate_cross_radar_candidate` |
| `A090` | `KEMX_T006;KIWA_T003` | 8.8 | 17.25 km | 290 m | `moderate_cross_radar_candidate` |
| `A091` | `KEMX_T006;KIWA_T005` | 8.8 | 17.19 km | 290 m | `moderate_cross_radar_candidate` |
| `A092` | `KEMX_T001;KIWA_T007` | 23.1 | 18.89 km | 223 m | `moderate_cross_radar_candidate` |
| `A093` | `KEMX_T006;KIWA_T009` | 8.8 | 17.25 km | 290 m | `moderate_cross_radar_candidate` |
| `A094` | `KEMX_T001;KIWA_T004` | 30.1 | 19.78 km | 175 m | `moderate_cross_radar_candidate` |
| `A095` | `KEMX_T006;KIWA_T006` | 8.8 | 17.24 km | 316 m | `moderate_cross_radar_candidate` |
| `A096` | `KEMX_T006;KIWA_T004` | 30.1 | 20.61 km | 174 m | `moderate_cross_radar_candidate` |
| `A097` | `KEMX_T007;KIWA_T004` | 30.1 | 21.63 km | 173 m | `moderate_cross_radar_candidate` |
| `A098` | `KEMX_T006;KIWA_T007` | 23.1 | 22.02 km | 164 m | `moderate_cross_radar_candidate` |
| `A099` | `KEMX_T006;KIWA_T010` | 23.1 | 22.02 km | 164 m | `moderate_cross_radar_candidate` |
| `A100` | `KEMX_T006;KIWA_T008` | 23.1 | 22.02 km | 164 m | `moderate_cross_radar_candidate` |

## Telemetry Comparisons

| Tracklet ID | Overlap (min) | Mean Dist Corridor | Median Alt Diff | Speed Ratio | Match Label |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `KEMX_T008` | 35.2 | 3.7 km | 29 m | 0.94x | `telemetry_consistent_candidate` |
| `KEMX_T003` | 35.2 | 3.8 km | 29 m | 0.94x | `telemetry_consistent_candidate` |
| `KEMX_T007` | 63.6 | 4.1 km | 39 m | 1.45x | `telemetry_consistent_candidate` |
| `KEMX_T004` | 56.5 | 5.0 km | 46 m | 1.09x | `telemetry_consistent_candidate` |
| `KEMX_T010` | 35.2 | 5.8 km | 64 m | 1.6x | `telemetry_consistent_candidate` |
| `KEMX_T001` | 35.2 | 3.9 km | 73 m | 1.3x | `telemetry_consistent_candidate` |
| `KIWA_T004` | 56.9 | 9.4 km | 78 m | 1.09x | `telemetry_consistent_candidate` |
| `KEMX_T006` | 35.2 | 6.8 km | 88 m | 1.3x | `telemetry_consistent_candidate` |
| `KEMX_T002` | 35.2 | 4.3 km | 93 m | 1.02x | `telemetry_consistent_candidate` |
| `KEMX_T005` | 35.2 | 4.0 km | 93 m | 1.02x | `telemetry_consistent_candidate` |
| `KEMX_T009` | 35.2 | 4.1 km | 113 m | 1.02x | `telemetry_consistent_candidate` |
| `KIWA_T006` | 35.6 | 4.4 km | 126 m | 0.97x | `telemetry_consistent_candidate` |
| `KIWA_T007` | 49.8 | 7.3 km | 148 m | 1.06x | `telemetry_consistent_candidate` |
| `KIWA_T008` | 49.8 | 8.8 km | 148 m | 1.3x | `telemetry_consistent_candidate` |
| `KIWA_T001` | 35.6 | 4.5 km | 176 m | 0.73x | `telemetry_consistent_candidate` |
| `KIWA_T002` | 35.6 | 6.2 km | 176 m | 1.56x | `telemetry_consistent_candidate` |
| `KIWA_T005` | 35.6 | 6.1 km | 176 m | 1.45x | `telemetry_consistent_candidate` |
| `KIWA_T009` | 35.6 | 6.2 km | 177 m | 1.56x | `telemetry_consistent_candidate` |
| `KIWA_T003` | 35.6 | 4.5 km | 177 m | 0.97x | `telemetry_consistent_candidate` |
| `KIWA_T010` | 49.8 | 7.3 km | 184 m | 1.03x | `telemetry_consistent_candidate` |

## Scientific Conclusion

1. **Corridor Coverage:** Candidate returns were restricted to a 40 km wide horizontal band around the estimated balloon track. The vertical filter restricted analysis to ±1500 m of the telemetry altitude.
2. **Altitude Match:** Multiple tracklets in both KEMX and KIWA show excellent vertical agreement (median mismatch < 300 m) with the expected balloon telemetry.
3. **Cross-Radar Support:** Cross-radar association indicates that candidates from KEMX and KIWA are spatially and temporally compatible, suggesting they could represent the same physical target.
4. **Exact GPS Caveat:** Because the balloon horizontal telemetry comes from Maidenhead grid squares, the exact GPS track is uncertain. We cannot make positive claims of detected balloons or confirmed tracks.

---
*Report generated automatically by PicoCAST regional discovery pipeline.*