#!/usr/bin/env python
"""Build Evidence Audit v1 for the top review-packet candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.candidate_utils import load_config  # noqa: E402
from scripts.make_review_packet_dashboard import parse_ids  # noqa: E402

def clean_float(value, default=np.nan):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def generate_html_index(audit_df: pd.DataFrame, out_dir: Path, case_id: str):
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        f'    <title>Evidence Audit v1 - {case_id}</title>',
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background: #f9fafb; color: #111827; }",
        "        h1 { color: #1f2937; }",
        "        table { border-collapse: collapse; width: 100%; margin-top: 2rem; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "        th, td { text-align: left; padding: 0.75rem; border-bottom: 1px solid #e5e7eb; }",
        "        th { background: #f3f4f6; font-weight: 600; }",
        "        tr:hover { background: #f9fafb; }",
        "        .tag { padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600; display: inline-block; margin-bottom: 0.25rem; }",
        "        .tag-red { background: #fee2e2; color: #991b1b; }",
        "        .tag-green { background: #dcfce7; color: #166534; }",
        "        .tag-yellow { background: #fef9c3; color: #854d0e; }",
        "        .tag-gray { background: #f3f4f6; color: #374151; }",
        "    </style>",
        "</head>",
        "<body>",
        f"    <h1>Evidence Audit v1: {case_id}</h1>",
        "    <p>This audit summarizes the top 10 radar tracklet candidates from the review packet.</p>",
        "    <table>",
        "        <thead>",
        "            <tr>",
        "                <th>Rank</th>",
        "                <th>Candidate ID</th>",
        "                <th>Radar(s)</th>",
        "                <th>Verdict</th>",
        "                <th>Flags</th>",
        "                <th>Details</th>",
        "            </tr>",
        "        </thead>",
        "        <tbody>"
    ]

    for _, row in audit_df.iterrows():
        flags = []
        if row["weather_contamination"]: flags.append("Weather")
        if row["ground_clutter"]: flags.append("Clutter")
        if row["biological_echo"]: flags.append("Biology")
        if row["anomalous_propagation"]: flags.append("AP")
        if row["isolated_artifact"]: flags.append("Isolated")
        if row["impossible_jump"]: flags.append("Jump")
        
        flags_html = "".join([f'<span class="tag tag-red">{f}</span> ' for f in flags])
        if not flags:
            flags_html = '<span class="tag tag-green">Clean</span>'

        verdict_color = "gray"
        if row["final_verdict"] == "strong candidate": verdict_color = "green"
        elif row["final_verdict"] == "plausible but weak": verdict_color = "yellow"
        elif row["final_verdict"] == "likely false positive": verdict_color = "red"
        
        lines.extend([
            "            <tr>",
            f"                <td>{row['review_rank']}</td>",
            f"                <td><strong>{row['item_id']}</strong></td>",
            f"                <td>{row['radar_sites']}</td>",
            f"                <td><span class=\"tag tag-{verdict_color}\">{row['final_verdict']}</span></td>",
            f"                <td>{flags_html}</td>",
            f"                <td><a href=\"candidate_{row['item_id']}.html\">View Evidence</a></td>",
            "            </tr>"
        ])

    lines.extend([
        "        </tbody>",
        "    </table>",
        "</body>",
        "</html>"
    ])
    
    (out_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")

def generate_candidate_html(row: pd.Series, out_dir: Path, case_id: str):
    cid = row["item_id"]
    
    # We will grab the plot image from the review_packet/plots directory.
    # The plot path in the CSV usually looks like 'plots/review_rank_01_A001.png'
    # We are in outputs/discovery/evidence_audit/
    # the image is in outputs/discovery/review_packet/plots/
    # So relative path: ../review_packet/plots/review_rank_XX_YYY.png
    
    orig_plot_path = str(row.get("orig_plot_path", ""))
    img_src = f"../review_packet/{orig_plot_path}" if orig_plot_path else ""

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '    <meta charset="UTF-8">',
        f'    <title>Candidate {cid} - Evidence Audit</title>',
        "    <style>",
        "        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 2rem; background: #f9fafb; color: #111827; }",
        "        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        "        h1 { color: #1f2937; margin-top: 0; }",
        "        h2 { border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; margin-top: 2rem; }",
        "        table { border-collapse: collapse; width: 100%; margin-top: 1rem; }",
        "        th, td { text-align: left; padding: 0.75rem; border-bottom: 1px solid #e5e7eb; }",
        "        th { background: #f3f4f6; width: 30%; }",
        "        .plot-container { text-align: center; margin-top: 2rem; background: #f3f4f6; padding: 1rem; border-radius: 8px; }",
        "        .plot-container img { max-width: 100%; height: auto; border-radius: 4px; }",
        "        .back-link { display: inline-block; margin-bottom: 1rem; color: #2563eb; text-decoration: none; font-weight: 500; }",
        "        .back-link:hover { text-decoration: underline; }",
        "        .tag { padding: 0.25rem 0.5rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; display: inline-block; margin-bottom: 0.25rem; margin-right: 0.5rem; }",
        "        .tag-red { background: #fee2e2; color: #991b1b; }",
        "        .tag-green { background: #dcfce7; color: #166534; }",
        "    </style>",
        "</head>",
        "<body>",
        "    <div class=\"container\">",
        "        <a href=\"index.html\" class=\"back-link\">&larr; Back to Audit Index</a>",
        f"        <h1>Evidence Audit: Candidate {cid}</h1>",
        "        <h2>Summary</h2>",
        "        <table>",
        "            <tbody>",
        f"                <tr><th>Verdict</th><td><strong>{row['final_verdict']}</strong></td></tr>",
        f"                <tr><th>Rank</th><td>{row['review_rank']}</td></tr>",
        f"                <tr><th>Radar Site(s)</th><td>{row['radar_sites']}</td></tr>",
        f"                <tr><th>Type</th><td>{row['type_label']}</td></tr>",
        f"                <tr><th>Time Range</th><td>{row['time_range']}</td></tr>",
        f"                <tr><th>Number of Scans</th><td>{row['n_scans']}</td></tr>",
        "            </tbody>",
        "        </table>",
        "",
        "        <h2>Radar & Telemetry Evidence</h2>",
        "        <table>",
        "            <tbody>",
        f"                <tr><th>Altitude Agreement (Median)</th><td>{row['altitude_agreement_m']:.1f} m</td></tr>",
        f"                <tr><th>Horizontal Offset (Median)</th><td>{row['horizontal_offset_km']:.2f} km</td></tr>",
        f"                <tr><th>Motion Consistency (Std Dev Velocity)</th><td>{row['motion_consistency_ms']:.2f} m/s</td></tr>",
        f"                <tr><th>Reflectivity (Mean / Max)</th><td>{row['reflectivity_mean_dbz']:.1f} / {row['reflectivity_max_dbz']:.1f} dBZ</td></tr>",
        f"                <tr><th>Velocity (Mean / Max)</th><td>{row['velocity_mean_ms']:.1f} / {row['velocity_max_ms']:.1f} m/s</td></tr>",
        f"                <tr><th>Spectrum Width (Mean / Max)</th><td>{row['spectrum_width_mean_ms']:.1f} / {row['spectrum_width_max_ms']:.1f} m/s</td></tr>",
        "            </tbody>",
        "        </table>",
        "",
        "        <h2>False Positive Flags</h2>",
        "        <div>"
    ]
    
    flags = [
        ("Weather Contamination", row["weather_contamination"]),
        ("Ground Clutter / Low Elevation", row["ground_clutter"]),
        ("Biological Echo Risk", row["biological_echo"]),
        ("Anomalous Propagation Risk", row["anomalous_propagation"]),
        ("Isolated Artifact", row["isolated_artifact"]),
        ("Impossible Jump", row["impossible_jump"]),
    ]
    
    for flag_name, is_flagged in flags:
        if is_flagged:
            lines.append(f'            <span class="tag tag-red">&#10008; {flag_name}</span>')
        else:
            lines.append(f'            <span class="tag tag-green">&#10004; {flag_name}</span>')

    lines.extend([
        "        </div>",
        "",
        "        <h2>Visual Evidence</h2>",
        "        <div class=\"plot-container\">"
    ])
    
    if img_src:
        lines.append(f"            <img src=\"{img_src}\" alt=\"Candidate Plot\" />")
    else:
        lines.append("            <p>No plot available for this candidate.</p>")
        
    lines.extend([
        "        </div>",
        "    </div>",
        "</body>",
        "</html>"
    ])
    
    (out_dir / f"candidate_{cid}.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to case config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    case_dir = args.config.parent
    out_dir = case_dir / "outputs" / "discovery" / "evidence_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    review_queue_path = case_dir / "outputs" / "discovery" / "review_packet" / "tracklet_review_queue.csv"
    points_path = case_dir / "outputs" / "discovery" / "plausible_tracklet_points.csv"
    
    if not review_queue_path.exists() or not points_path.exists():
        print(f"Missing required CSVs in {case_dir}")
        return

    queue_df = pd.read_csv(review_queue_path)
    points_df = pd.read_csv(points_path)
    
    # Take top 10
    top_10 = queue_df.head(10).copy()
    
    audit_rows = []
    
    for idx, row in top_10.iterrows():
        t_ids = parse_ids(row["tracklet_ids"])
        pts = points_df[points_df["tracklet_id"].isin(t_ids)].copy()
        
        if pts.empty:
            continue
            
        pts = pts.sort_values("scan_time_utc").reset_index(drop=True)
        
        # Calculate metrics
        times = pd.to_datetime(pts["scan_time_utc"], utc=True)
        time_range = f"{times.min().strftime('%H:%M')} - {times.max().strftime('%H:%M')} UTC"
        n_scans = len(pts)
        
        alt_agr = clean_float(pts["abs_vertical_distance_m"].median())
        horiz_off = clean_float(pts.get("distance_to_track_corridor_km", pd.Series([np.nan]*len(pts))).median())
        
        vel_mean = clean_float(pts["velocity_mean_ms"].mean())
        vel_max = clean_float(pts["velocity_mean_ms"].max())
        vel_std = clean_float(pts["velocity_mean_ms"].std()) if n_scans > 1 else 0.0
        
        ref_mean = clean_float(pts["mean_reflectivity_dbz"].mean())
        ref_max = clean_float(pts["mean_reflectivity_dbz"].max())
        
        sw_mean = clean_float(pts["spectrum_width_mean_ms"].mean()) if "spectrum_width_mean_ms" in pts.columns else np.nan
        sw_max = clean_float(pts["spectrum_width_mean_ms"].max()) if "spectrum_width_mean_ms" in pts.columns else np.nan
        
        elev_min = clean_float(pts["elevation_deg"].min()) if "elevation_deg" in pts.columns else 999.0
        alt_min = clean_float(pts["cluster_alt_m"].min())
        
        # speeds between points
        speeds_kmh = []
        for i in range(1, len(pts)):
            dt_h = (times.iloc[i] - times.iloc[i-1]).total_seconds() / 3600.0
            from scripts.candidate_utils import horizontal_distance_km
            if dt_h > 0:
                dist = horizontal_distance_km(
                    pts.loc[i-1, "cluster_lat_deg"], pts.loc[i-1, "cluster_lon_deg"],
                    pts.loc[i, "cluster_lat_deg"], pts.loc[i, "cluster_lon_deg"]
                )
                speeds_kmh.append(dist / dt_h)
        max_speed = max(speeds_kmh) if speeds_kmh else 0.0

        # Heuristics
        weather_contamination = bool(ref_mean > 15.0)
        ground_clutter = bool(elev_min < 1.0 or alt_min < 2000)
        biological_echo = bool(ref_mean < 10.0 and alt_min < 3000)
        anomalous_propagation = bool(abs(vel_mean) < 2.0 and ref_mean > 10.0)
        isolated_artifact = bool(n_scans == 1)
        impossible_jump = bool(max_speed > 150.0)
        
        # Final verdict
        score = clean_float(row["review_priority_score"])
        if score > 0.8:
            verdict = "strong candidate"
        elif score > 0.7:
            verdict = "plausible but weak"
        else:
            verdict = "likely false positive"
            
        audit_rows.append({
            "review_rank": row["review_rank"],
            "item_id": row["item_id"],
            "radar_sites": row["radar_sites"],
            "type_label": "cross-radar" if "cross" in str(row["item_type"]).lower() else "single-radar",
            "time_range": time_range,
            "n_scans": n_scans,
            "altitude_agreement_m": alt_agr,
            "horizontal_offset_km": horiz_off,
            "motion_consistency_ms": vel_std,
            "reflectivity_mean_dbz": ref_mean,
            "reflectivity_max_dbz": ref_max,
            "velocity_mean_ms": vel_mean,
            "velocity_max_ms": vel_max,
            "spectrum_width_mean_ms": sw_mean,
            "spectrum_width_max_ms": sw_max,
            "weather_contamination": weather_contamination,
            "ground_clutter": ground_clutter,
            "biological_echo": biological_echo,
            "anomalous_propagation": anomalous_propagation,
            "isolated_artifact": isolated_artifact,
            "impossible_jump": impossible_jump,
            "final_verdict": verdict,
            "orig_plot_path": row.get("plot_path", "")
        })

    audit_df = pd.DataFrame(audit_rows)
    csv_path = out_dir / "top_tracklet_audit.csv"
    
    # Save CSV
    export_df = audit_df.drop(columns=["orig_plot_path"])
    export_df.to_csv(csv_path, index=False)
    print(f"Wrote audit CSV: {csv_path}")
    
    # Generate HTML
    generate_html_index(audit_df, out_dir, case_id=config["case_id"])
    print(f"Wrote index HTML: {out_dir / 'index.html'}")
    
    for _, row in audit_df.iterrows():
        generate_candidate_html(row, out_dir, case_id=config["case_id"])

if __name__ == "__main__":
    main()
