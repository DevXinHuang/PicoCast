from nexrad_picoballoon.detect import detect_candidates, score_gates
from nexrad_picoballoon.features import extract_gate_features
from nexrad_picoballoon.synthetic import make_synthetic_volume
from nexrad_picoballoon.tracking import build_tracks


def test_detector_finds_synthetic_target():
    features = extract_gate_features(make_synthetic_volume())
    scored = score_gates(features)
    assert scored["candidate_score"].max() >= 0.65

    candidates = detect_candidates(features)
    assert len(candidates) == 1
    assert candidates.iloc[0]["gate_count"] > 0
    assert candidates.iloc[0]["score"] >= 0.65


def test_tracking_builds_site_track_from_candidates():
    features = extract_gate_features(make_synthetic_volume())
    candidates = detect_candidates(features)
    tracks = build_tracks(candidates)

    assert len(tracks) == 1
    assert tracks.iloc[0]["site"] == "KTEST"
    assert tracks.iloc[0]["continuity_score"] > 0
