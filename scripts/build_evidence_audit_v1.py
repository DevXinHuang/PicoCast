#!/usr/bin/env python
"""Build Evidence Audit v1.1 for the top review-packet candidates.

v1.1 change: Final verdict is now driven by an independent evidence_score
computed from audit metrics only (altitude, offset, motion, scans, radar
moments, cross-radar support, false-positive flags). The original
review_priority_score is preserved in the output but is no longer used
as the verdict source, avoiding circular reasoning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import horizontal_distance_km, load_config  # noqa: E402
from scripts.make_review_packet_dashboard import parse_ids  # noqa: E402

AUDIT_VERSION = "v1.1"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def clean_float(value, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def verdict_color(verdict: str) -> str:
    return {
        "strong candidate": "green",
        "plausible but weak": "yellow",
        "likely false positive": "red",
        "reject": "red",
    }.get(verdict, "gray")


def v1_verdict_from_score(score: float) -> str:
    """Original v1 verdict using review_priority_score."""
    if score > 0.80:
        return "strong candidate"
    elif score > 0.70:
        return "plausible but weak"
    return "likely false positive"


# ─── Independent evidence scoring ────────────────────────────────────────────

def compute_evidence_score(
    *,
    altitude_agreement_m: float,
    horizontal_offset_km: float,
    motion_consistency_ms: float,
    n_scans: int,
    reflectivity_mean_dbz: float,
    velocity_mean_ms: float,
    spectrum_width_mean_ms: float,
    is_cross_radar: bool,
    weather_contamination: bool,
    ground_clutter: bool,
    biological_echo: bool,
    anomalous_propagation: bool,
    isolated_artifact: bool,
    impossible_jump: bool,
) -> tuple[int, int, int, str]:
    """Return (positive_count, negative_count, evidence_score, reason_text)."""
    positives: list[str] = []
    negatives: list[str] = []

    # --- Positive evidence ---
    if np.isfinite(altitude_agreement_m) and altitude_agreement_m < 150:
        positives.append(f"altitude agreement {altitude_agreement_m:.0f} m < 150 m threshold")
    if np.isfinite(horizontal_offset_km) and horizontal_offset_km < 15:
        positives.append(f"horizontal offset {horizontal_offset_km:.1f} km < 15 km threshold")
    if np.isfinite(motion_consistency_ms) and motion_consistency_ms < 3.0:
        positives.append(f"motion consistency (σ={motion_consistency_ms:.2f} m/s) < 3.0 m/s")
    if n_scans >= 5:
        positives.append(f"persists across {n_scans} scans (≥5 required)")
    if np.isfinite(reflectivity_mean_dbz) and -15 <= reflectivity_mean_dbz <= 5:
        positives.append(f"reflectivity {reflectivity_mean_dbz:.1f} dBZ in balloon-plausible range [−15, +5]")
    if np.isfinite(velocity_mean_ms) and abs(velocity_mean_ms) < 30:
        positives.append(f"velocity {velocity_mean_ms:.1f} m/s within plausible drift range (|v| < 30 m/s)")
    if np.isfinite(spectrum_width_mean_ms) and spectrum_width_mean_ms < 4.0:
        positives.append(f"spectrum width {spectrum_width_mean_ms:.1f} m/s < 4.0 m/s (narrow, non-weather)")
    if is_cross_radar:
        positives.append("cross-radar corroboration from ≥2 independent sites")

    # --- Negative evidence (false-positive flags) ---
    if weather_contamination:
        negatives.append("weather contamination (reflectivity > 15 dBZ)")
    if ground_clutter:
        negatives.append("ground clutter / low-elevation risk")
    if biological_echo:
        negatives.append("biological echo risk (low reflectivity at low altitude)")
    if anomalous_propagation:
        negatives.append("anomalous propagation risk (low velocity, moderate reflectivity)")
    if isolated_artifact:
        negatives.append("isolated one-scan artifact (n_scans == 1)")
    if impossible_jump:
        negatives.append("impossible trajectory jump (speed > 150 km/h)")

    pos_count = len(positives)
    neg_count = len(negatives)
    score = pos_count - neg_count

    reason_parts = []
    if positives:
        reason_parts.append("Positive: " + "; ".join(positives))
    if negatives:
        reason_parts.append("Negative: " + "; ".join(negatives))
    reason = " | ".join(reason_parts) if reason_parts else "No evidence either way"

    return pos_count, neg_count, score, reason


def v11_verdict(evidence_score: int, negative_flag_count: int) -> str:
    if evidence_score >= 4 and negative_flag_count == 0:
        return "strong candidate"
    elif evidence_score >= 2 and negative_flag_count <= 1:
        return "plausible but weak"
    return "likely false positive"


# ─── HTML generation ─────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  margin: 0; background: #f1f5f9; color: #0f172a;
}
.page { max-width: 1280px; margin: 0 auto; padding: 2rem; }
h1 { margin-top: 0; color: #0f172a; font-size: 1.6rem; }
h2 { font-size: 1.1rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 0.4rem;
     margin-top: 2rem; color: #1e293b; }
p { color: #475569; line-height: 1.6; }
a { color: #2563eb; text-decoration: none; }
a:hover { text-decoration: underline; }
table { border-collapse: collapse; width: 100%; margin-top: 1rem;
        background: white; border-radius: 8px; overflow: hidden;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
th, td { text-align: left; padding: 0.7rem 1rem; border-bottom: 1px solid #e2e8f0; }
th { background: #f8fafc; font-weight: 600; font-size: 0.82rem;
     text-transform: uppercase; letter-spacing: 0.04em; color: #64748b; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #f8fafc; }
.card { background: white; border-radius: 8px; padding: 1.5rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-top: 1.5rem; }
.tag { padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.75rem;
       font-weight: 700; display: inline-block; margin: 0.15rem; }
.tag-green  { background: #dcfce7; color: #15803d; }
.tag-yellow { background: #fef9c3; color: #a16207; }
.tag-red    { background: #fee2e2; color: #b91c1c; }
.tag-blue   { background: #dbeafe; color: #1d4ed8; }
.tag-gray   { background: #f1f5f9; color: #475569; }
.score-box { display: inline-block; border-radius: 8px; padding: 0.5rem 1.2rem;
             font-size: 1.5rem; font-weight: 800; margin-right: 1rem; }
.score-green  { background: #dcfce7; color: #15803d; }
.score-yellow { background: #fef9c3; color: #a16207; }
.score-red    { background: #fee2e2; color: #b91c1c; }
.score-gray   { background: #f1f5f9; color: #475569; }
.change-badge { font-size: 0.8rem; font-weight: 700; padding: 0.2rem 0.6rem;
                border-radius: 4px; margin-left: 0.5rem; }
.upgraded   { background: #dcfce7; color: #15803d; }
.downgraded { background: #fee2e2; color: #b91c1c; }
.unchanged  { background: #f1f5f9; color: #64748b; }
.flag-row   { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.75rem; }
.reason-box { background: #f8fafc; border-left: 3px solid #94a3b8; padding: 0.75rem 1rem;
              font-size: 0.85rem; color: #334155; border-radius: 0 6px 6px 0;
              margin-top: 0.75rem; line-height: 1.7; }
.plot-box { text-align: center; background: #f8fafc; border-radius: 8px;
            padding: 1rem; margin-top: 1rem; }
.plot-box img { max-width: 100%; border-radius: 6px; }
.back-link { display: inline-block; margin-bottom: 1.5rem; color: #2563eb;
             font-weight: 500; }
.scores-row { display: flex; gap: 1.5rem; align-items: flex-start;
              flex-wrap: wrap; margin-top: 0.75rem; }
.score-item { text-align: center; }
.score-label { font-size: 0.75rem; color: #64748b; font-weight: 600;
               text-transform: uppercase; letter-spacing: 0.04em;
               margin-bottom: 0.3rem; }
"""


def verdict_score_class(verdict: str) -> str:
    return {
        "strong candidate": "score-green",
        "plausible but weak": "score-yellow",
        "likely false positive": "score-red",
        "reject": "score-red",
    }.get(verdict, "score-gray")


def change_badge_html(changed: bool, v11: str, v1: str) -> str:
    if not changed:
        return '<span class="change-badge unchanged">Unchanged from v1</span>'
    verdicts = ["likely false positive", "plausible but weak", "strong candidate"]
    try:
        up = verdicts.index(v11) > verdicts.index(v1)
    except ValueError:
        up = False
    if up:
        return f'<span class="change-badge upgraded">&#8593; Upgraded from v1: was "{v1}"</span>'
    return f'<span class="change-badge downgraded">&#8595; Downgraded from v1: was "{v1}"</span>'


def generate_html_index(audit_df: pd.DataFrame, out_dir: Path, case_id: str) -> None:
    rows_html = []
    for _, row in audit_df.iterrows():
        flags = []
        if row["weather_contamination"]: flags.append("Weather")
        if row["ground_clutter"]: flags.append("Clutter")
        if row["biological_echo"]: flags.append("Biology")
        if row["anomalous_propagation"]: flags.append("AP")
        if row["isolated_artifact"]: flags.append("Isolated")
        if row["impossible_jump"]: flags.append("Jump")

        flags_html = "".join(
            f'<span class="tag tag-red">{f}</span>' for f in flags
        ) or '<span class="tag tag-green">Clean</span>'

        vc = verdict_color(row["verdict_v1_1"])
        changed = bool(row["verdict_changed_from_v1"])
        change_icon = ""
        if changed:
            verdicts = ["likely false positive", "plausible but weak", "strong candidate"]
            try:
                up = verdicts.index(row["verdict_v1_1"]) > verdicts.index(row["verdict_v1"])
            except ValueError:
                up = False
            change_icon = ' <span style="color:#15803d">&#8593;</span>' if up else \
                          ' <span style="color:#b91c1c">&#8595;</span>'

        evs = int(row["evidence_score"])
        esc = verdict_score_class(row["verdict_v1_1"])
        rps = clean_float(row["review_priority_score"])

        rows_html.append(f"""
            <tr>
              <td>{row['review_rank']}</td>
              <td><strong>{row['item_id']}</strong><br>
                  <small style="color:#64748b">{row['type_label']}</small></td>
              <td>{row['radar_sites']}</td>
              <td><span class="tag tag-{vc}">{row['verdict_v1_1']}</span>{change_icon}</td>
              <td><span class="score-box {esc}" style="font-size:1rem;padding:0.2rem 0.7rem">{evs:+d}</span>
                  <small style="color:#94a3b8">({row['positive_evidence_count']}+ / {row['negative_flag_count']}-)</small></td>
              <td style="color:#64748b;font-size:0.88rem">{rps:.3f}</td>
              <td>{flags_html}</td>
              <td><a href="candidate_{row['item_id']}.html">View &rarr;</a></td>
            </tr>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evidence Audit {AUDIT_VERSION} — {case_id}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
  <h1>Evidence Audit {AUDIT_VERSION} — {case_id}</h1>
  <p>
    Top 10 review-packet candidates evaluated using an <strong>independent evidence score</strong>
    computed from radar moments, telemetry agreement, persistence, and false-positive flags —
    not from the detector's own ranking. The original <code>review_priority_score</code> is shown
    for reference only.
  </p>
  <table>
    <thead>
      <tr>
        <th>Rank</th>
        <th>Candidate</th>
        <th>Radar(s)</th>
        <th>Verdict v1.1</th>
        <th>Evidence Score</th>
        <th>Priority Score (v1 ref)</th>
        <th>FP Flags</th>
        <th>Details</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
  <p style="margin-top:2rem;font-size:0.8rem;color:#94a3b8">
    Generated by PicoCAST {AUDIT_VERSION} Evidence Audit pipeline.
    &#8593;/&#8595; indicates verdict upgrade/downgrade relative to v1.
  </p>
</div>
</body>
</html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def generate_candidate_html(row: pd.Series, out_dir: Path, case_id: str) -> None:
    cid = row["item_id"]
    orig_plot_path = str(row.get("orig_plot_path", ""))
    img_src = f"../review_packet/{orig_plot_path}" if orig_plot_path else ""

    vc_11 = verdict_color(row["verdict_v1_1"])
    vc_1  = verdict_color(row["verdict_v1"])
    esc   = verdict_score_class(row["verdict_v1_1"])
    evs   = int(row["evidence_score"])
    rps   = clean_float(row["review_priority_score"])
    changed = bool(row["verdict_changed_from_v1"])
    change_badge = change_badge_html(changed, row["verdict_v1_1"], row["verdict_v1"])

    # Build flag rows
    flags_def = [
        ("Weather Contamination", "weather_contamination"),
        ("Ground Clutter / Low Elevation", "ground_clutter"),
        ("Biological Echo Risk", "biological_echo"),
        ("Anomalous Propagation Risk", "anomalous_propagation"),
        ("Isolated Artifact (1-scan)", "isolated_artifact"),
        ("Impossible Trajectory Jump", "impossible_jump"),
    ]
    flags_html_parts = []
    for label, col in flags_def:
        if row[col]:
            flags_html_parts.append(f'<span class="tag tag-red">&#10008; {label}</span>')
        else:
            flags_html_parts.append(f'<span class="tag tag-green">&#10004; {label}</span>')

    # Reason — split positive / negative
    reason_raw = str(row.get("evidence_score_reason", ""))
    reason_parts = reason_raw.split(" | ")
    reason_html = ""
    for part in reason_parts:
        if part.startswith("Positive:"):
            items = part[len("Positive:"):].strip().split("; ")
            reason_html += "<strong style='color:#15803d'>Positive evidence:</strong><ul style='margin:0.3rem 0 0.6rem 1.2rem;padding:0'>"
            for it in items:
                reason_html += f"<li>{it}</li>"
            reason_html += "</ul>"
        elif part.startswith("Negative:"):
            items = part[len("Negative:"):].strip().split("; ")
            reason_html += "<strong style='color:#b91c1c'>Negative evidence (flags):</strong><ul style='margin:0.3rem 0 0.6rem 1.2rem;padding:0'>"
            for it in items:
                reason_html += f"<li>{it}</li>"
            reason_html += "</ul>"

    sw_mean = clean_float(row["spectrum_width_mean_ms"])
    sw_max  = clean_float(row["spectrum_width_max_ms"])
    sw_str  = f"{sw_mean:.1f} / {sw_max:.1f} m/s" if np.isfinite(sw_mean) else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Candidate {cid} — Evidence Audit {AUDIT_VERSION}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
  <a class="back-link" href="index.html">&larr; Back to Audit Index</a>
  <h1>Candidate {cid}</h1>

  <!-- ── Verdict & Score Breakdown ── -->
  <div class="card">
    <h2 style="margin-top:0">Verdict &amp; Evidence Score Breakdown</h2>
    <div class="scores-row">
      <div class="score-item">
        <div class="score-label">Verdict v1.1 (independent)</div>
        <span class="tag tag-{vc_11}" style="font-size:1rem;padding:0.4rem 1rem">
          {row['verdict_v1_1']}
        </span>
        {change_badge}
      </div>
      <div class="score-item">
        <div class="score-label">Evidence Score</div>
        <span class="score-box {esc}">{evs:+d}</span>
        <small style="color:#64748b">{row['positive_evidence_count']}+ &nbsp; {row['negative_flag_count']}−</small>
      </div>
      <div class="score-item">
        <div class="score-label">Review Priority Score (v1 ref only)</div>
        <span class="score-box score-gray" style="font-size:1rem;padding:0.4rem 1rem">
          {rps:.3f}
        </span>
        <span class="tag tag-{vc_1}" style="font-size:0.75rem">v1: {row['verdict_v1']}</span>
      </div>
    </div>
    <div class="reason-box">
      {reason_html if reason_html else reason_raw}
    </div>
  </div>

  <!-- ── Identification ── -->
  <div class="card">
    <h2 style="margin-top:0">Identification</h2>
    <table>
      <tbody>
        <tr><th style="width:30%">Rank</th><td>{row['review_rank']}</td></tr>
        <tr><th>Candidate ID</th><td><strong>{cid}</strong></td></tr>
        <tr><th>Type</th><td>{row['type_label']}</td></tr>
        <tr><th>Radar Site(s)</th><td>{row['radar_sites']}</td></tr>
        <tr><th>Time Range</th><td>{row['time_range']}</td></tr>
        <tr><th>Number of Scans</th><td>{row['n_scans']}</td></tr>
      </tbody>
    </table>
  </div>

  <!-- ── Radar & Telemetry Metrics ── -->
  <div class="card">
    <h2 style="margin-top:0">Radar &amp; Telemetry Metrics</h2>
    <table>
      <tbody>
        <tr><th style="width:40%">Altitude Agreement (Median)</th><td>{clean_float(row['altitude_agreement_m']):.1f} m</td></tr>
        <tr><th>Horizontal Offset (Median)</th><td>{clean_float(row['horizontal_offset_km']):.2f} km</td></tr>
        <tr><th>Motion Consistency (Velocity σ)</th><td>{clean_float(row['motion_consistency_ms']):.2f} m/s</td></tr>
        <tr><th>Reflectivity (Mean / Max)</th><td>{clean_float(row['reflectivity_mean_dbz']):.1f} / {clean_float(row['reflectivity_max_dbz']):.1f} dBZ</td></tr>
        <tr><th>Velocity (Mean / Max)</th><td>{clean_float(row['velocity_mean_ms']):.1f} / {clean_float(row['velocity_max_ms']):.1f} m/s</td></tr>
        <tr><th>Spectrum Width (Mean / Max)</th><td>{sw_str}</td></tr>
      </tbody>
    </table>
  </div>

  <!-- ── False Positive Flags ── -->
  <div class="card">
    <h2 style="margin-top:0">False-Positive Flags</h2>
    <div class="flag-row">
      {"".join(flags_html_parts)}
    </div>
  </div>

  <!-- ── Visual Evidence ── -->
  <div class="card">
    <h2 style="margin-top:0">Visual Evidence</h2>
    <div class="plot-box">
      {"<img src='" + img_src + "' alt='Candidate plot — map and altitude-time plot' />" if img_src else "<p style='color:#94a3b8'>No plot available for this candidate.</p>"}
    </div>
  </div>

  <p style="font-size:0.8rem;color:#94a3b8;margin-top:2rem">
    PicoCAST Evidence Audit {AUDIT_VERSION} — {case_id}
  </p>
</div>
</body>
</html>"""
    (out_dir / f"candidate_{cid}.html").write_text(html, encoding="utf-8")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    case_dir = args.config.parent
    out_dir = case_dir / "outputs" / "discovery" / "evidence_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    review_queue_path = (
        case_dir / "outputs" / "discovery" / "review_packet" / "tracklet_review_queue.csv"
    )
    points_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"

    if not review_queue_path.exists() or not points_path.exists():
        print(f"ERROR: Missing required CSVs under {case_dir}")
        return

    queue_df  = pd.read_csv(review_queue_path)
    points_df = pd.read_csv(points_path)
    top_10    = queue_df.head(10).copy()

    audit_rows: list[dict] = []

    for _, row in top_10.iterrows():
        t_ids = parse_ids(row["tracklet_ids"])
        pts   = points_df[points_df["tracklet_id"].isin(t_ids)].copy()
        if pts.empty:
            continue

        pts   = pts.sort_values("scan_time_utc").reset_index(drop=True)
        times = pd.to_datetime(pts["scan_time_utc"], utc=True)

        # ── Base metrics ──────────────────────────────────────────────────────
        time_range  = f"{times.min().strftime('%H:%M')} – {times.max().strftime('%H:%M')} UTC"
        n_scans     = len(pts)
        alt_agr     = clean_float(pts["abs_vertical_distance_m"].median())
        horiz_off   = clean_float(
            pts.get("distance_to_track_corridor_km",
                    pd.Series([np.nan] * len(pts))).median()
        )
        vel_mean    = clean_float(pts["velocity_mean_ms"].mean())
        vel_max     = clean_float(pts["velocity_mean_ms"].max())
        vel_std     = clean_float(pts["velocity_mean_ms"].std()) if n_scans > 1 else 0.0
        ref_mean    = clean_float(pts["mean_reflectivity_dbz"].mean())
        ref_max     = clean_float(pts["mean_reflectivity_dbz"].max())
        sw_mean     = (
            clean_float(pts["spectrum_width_mean_ms"].mean())
            if "spectrum_width_mean_ms" in pts.columns else np.nan
        )
        sw_max      = (
            clean_float(pts["spectrum_width_mean_ms"].max())
            if "spectrum_width_mean_ms" in pts.columns else np.nan
        )
        elev_min    = (
            clean_float(pts["elevation_deg"].min())
            if "elevation_deg" in pts.columns else 999.0
        )
        alt_min     = clean_float(pts["cluster_alt_m"].min())

        # ── Speed between consecutive points ─────────────────────────────────
        max_speed_kmh = 0.0
        for i in range(1, len(pts)):
            dt_h = (times.iloc[i] - times.iloc[i - 1]).total_seconds() / 3600.0
            if dt_h > 0:
                dist = horizontal_distance_km(
                    pts.loc[i - 1, "cluster_lat_deg"], pts.loc[i - 1, "cluster_lon_deg"],
                    pts.loc[i,     "cluster_lat_deg"], pts.loc[i,     "cluster_lon_deg"],
                )
                max_speed_kmh = max(max_speed_kmh, dist / dt_h)

        # ── False-positive flags (heuristics) ─────────────────────────────────
        weather_contamination = bool(ref_mean > 15.0)
        ground_clutter        = bool(elev_min < 1.0 or alt_min < 2000)
        biological_echo       = bool(ref_mean < 10.0 and alt_min < 3000)
        anomalous_propagation = bool(abs(vel_mean) < 2.0 and ref_mean > 10.0)
        isolated_artifact     = bool(n_scans == 1)
        impossible_jump       = bool(max_speed_kmh > 150.0)

        is_cross_radar = "cross" in str(row["item_type"]).lower()
        type_label     = "cross-radar" if is_cross_radar else "single-radar"

        # ── Independent evidence score (v1.1) ─────────────────────────────────
        pos_count, neg_count, ev_score, ev_reason = compute_evidence_score(
            altitude_agreement_m    = alt_agr,
            horizontal_offset_km    = horiz_off,
            motion_consistency_ms   = vel_std,
            n_scans                 = n_scans,
            reflectivity_mean_dbz   = ref_mean,
            velocity_mean_ms        = vel_mean,
            spectrum_width_mean_ms  = sw_mean,
            is_cross_radar          = is_cross_radar,
            weather_contamination   = weather_contamination,
            ground_clutter          = ground_clutter,
            biological_echo         = biological_echo,
            anomalous_propagation   = anomalous_propagation,
            isolated_artifact       = isolated_artifact,
            impossible_jump         = impossible_jump,
        )

        verdict_11 = v11_verdict(ev_score, neg_count)

        # v1 verdict uses review_priority_score (kept for comparison only)
        rps = clean_float(row["review_priority_score"])
        verdict_1 = v1_verdict_from_score(rps)

        audit_rows.append({
            # identification
            "review_rank":              row["review_rank"],
            "item_id":                  row["item_id"],
            "radar_sites":              row["radar_sites"],
            "type_label":               type_label,
            "time_range":               time_range,
            "n_scans":                  n_scans,
            # telemetry/geometry metrics
            "altitude_agreement_m":     alt_agr,
            "horizontal_offset_km":     horiz_off,
            "motion_consistency_ms":    vel_std,
            # radar moments
            "reflectivity_mean_dbz":    ref_mean,
            "reflectivity_max_dbz":     ref_max,
            "velocity_mean_ms":         vel_mean,
            "velocity_max_ms":          vel_max,
            "spectrum_width_mean_ms":   sw_mean,
            "spectrum_width_max_ms":    sw_max,
            # false-positive flags
            "weather_contamination":    weather_contamination,
            "ground_clutter":           ground_clutter,
            "biological_echo":          biological_echo,
            "anomalous_propagation":    anomalous_propagation,
            "isolated_artifact":        isolated_artifact,
            "impossible_jump":          impossible_jump,
            # v1.1 independent scoring
            "evidence_score":           ev_score,
            "evidence_score_reason":    ev_reason,
            "positive_evidence_count":  pos_count,
            "negative_flag_count":      neg_count,
            "verdict_v1_1":             verdict_11,
            # v1 comparison
            "review_priority_score":    rps,
            "verdict_v1":               verdict_1,
            "verdict_changed_from_v1":  verdict_11 != verdict_1,
            # internal — not exported to CSV
            "orig_plot_path":           row.get("plot_path", ""),
        })

    audit_df = pd.DataFrame(audit_rows)

    # ── Export CSV (drop internal column) ─────────────────────────────────────
    csv_path  = out_dir / "top_tracklet_audit.csv"
    export_df = audit_df.drop(columns=["orig_plot_path"])
    export_df.to_csv(csv_path, index=False)
    print(f"Wrote audit CSV  : {csv_path}")

    # ── Generate HTML ─────────────────────────────────────────────────────────
    generate_html_index(audit_df, out_dir, case_id=config["case_id"])
    print(f"Wrote index HTML : {out_dir / 'index.html'}")

    for _, row in audit_df.iterrows():
        generate_candidate_html(row, out_dir, case_id=config["case_id"])
        print(f"  candidate_{row['item_id']}.html")


if __name__ == "__main__":
    main()
