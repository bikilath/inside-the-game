"""
Tests for the analysis layer.

The generator is the input here, so these check that interpretation is
faithful: shares sum to 100, counts match a hand-count of the events, derived
rates agree with their own numerators and denominators, and the ranking and
state-classification logic behaves on inputs built to exercise it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from match_sim import MatchSim  # noqa: E402
import analyze  # noqa: E402


@pytest.fixture(scope="module")
def events_file(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("data") / "match.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for ev in MatchSim(seed=7).run():
            fh.write(json.dumps(ev) + "\n")
    return str(path)


@pytest.fixture(scope="module")
def report(events_file) -> dict:
    return analyze.build_report(events_file)


@pytest.fixture(scope="module")
def loaded(events_file):
    return analyze.load_events(events_file)


# ------------------------------------------------------------------ loading

def test_load_splits_metadata_from_events(loaded):
    meta, events = loaded
    assert meta["type"] == "match_metadata"
    assert all(e["type"] != "match_metadata" for e in events)
    assert len(events) > 1000


def test_team_names_resolve(loaded):
    meta, _ = loaded
    names = analyze.team_names(meta)
    assert set(names) == {"HOM", "AWY"}
    assert all(isinstance(v, str) and v for v in names.values())


# -------------------------------------------------------------- aggregation

def test_shares_sum_to_100(report):
    for key in ("possession_pct", "field_tilt_pct"):
        total = sum(report[key].values())
        assert total == pytest.approx(100.0, abs=0.2), f"{key} sums to {total}"


def test_pass_counts_match_a_hand_count(loaded, report):
    _, events = loaded
    for tid, stats in report["team_stats"].items():
        actual = sum(1 for e in events if e["type"] == "pass" and e["team_id"] == tid)
        assert stats["passes"] == actual


def test_pass_accuracy_agrees_with_its_own_counts(report):
    for stats in report["team_stats"].values():
        expected = 100 * stats["passes_completed"] / stats["passes"]
        assert stats["pass_accuracy"] == pytest.approx(expected, abs=0.05)


def test_completed_never_exceeds_attempted(report):
    for s in report["team_stats"].values():
        assert s["passes_completed"] <= s["passes"]
        assert s["progressive_passes"] <= s["passes_completed"]
        assert s["shots_on_target"] <= s["shots"]


def test_goals_in_stats_match_the_final_score(report):
    for tid, stats in report["team_stats"].items():
        assert stats["goals"] == report["final_score"][tid]


def test_xg_per_shot_is_consistent(report):
    for s in report["team_stats"].values():
        if s["shots"]:
            assert s["xg_per_shot"] == pytest.approx(s["xg"] / s["shots"], abs=0.002)


def test_raw_sample_lists_are_not_leaked(report):
    """Internal accumulators shouldn't reach the published report."""
    for s in report["team_stats"].values():
        assert "pass_difficulty" not in s
        assert "pressure_applied" not in s


# ------------------------------------------------------------------ players

def test_player_rows_are_sorted_by_involvement(report):
    touches = [p["touches"] for p in report["top_players"]]
    assert touches == sorted(touches, reverse=True)


def test_player_goals_sum_to_the_scoreline(loaded):
    _, events = loaded
    players = analyze.player_stats(events)
    assert sum(p["goals"] for p in players) == sum(
        1 for e in events if e["type"] == "shot" and e["outcome"] == "goal"
    )


def test_player_pass_accuracy_is_bounded(loaded):
    _, events = loaded
    for p in analyze.player_stats(events):
        assert 0.0 <= p["pass_accuracy"] <= 100.0
        assert p["passes_completed"] <= p["passes"]


# ----------------------------------------------------------------- momentum

def test_momentum_windows_are_contiguous_and_ordered(report):
    rows = report["momentum"]
    assert rows
    for i, r in enumerate(rows):
        assert r["from_min"] == i * 5
        assert r["to_min"] == r["from_min"] + 5


def test_each_window_splits_possession_to_100(report):
    for r in report["momentum"]:
        total = sum(r["possession_pct"].values())
        # a window with no on-ball events legitimately sums to 0
        assert total == pytest.approx(100.0, abs=0.2) or total == 0


def test_states_are_valid_labels(report):
    valid = {"controlled", "contested", "chaotic"}
    assert all(r["state"] in valid for r in report["momentum"])


def test_states_are_not_all_identical(report):
    """
    The original bug: absolute thresholds labelled all 19 windows 'chaotic'.
    Classification is relative to the match median, so a real match must vary.
    """
    states = {r["state"] for r in report["momentum"]}
    assert len(states) > 1, f"every window classified {states}"


def test_classifier_separates_calm_from_frantic():
    """Hand-built windows: the calm one must not outrank the frantic one."""
    rows = [
        {"from_min": 0, "to_min": 5, "tempo": 15, "turnovers_per_min": 1.0,
         "pressure_volatility": 0.05},
        {"from_min": 5, "to_min": 10, "tempo": 15, "turnovers_per_min": 2.0,
         "pressure_volatility": 0.15},
        {"from_min": 10, "to_min": 15, "tempo": 15, "turnovers_per_min": 6.0,
         "pressure_volatility": 0.40},
        {"from_min": 15, "to_min": 20, "tempo": 15, "turnovers_per_min": 7.0,
         "pressure_volatility": 0.45},
    ]
    analyze._classify_states(rows)
    assert rows[0]["state"] == "controlled"
    assert rows[-1]["state"] == "chaotic"


def test_classifier_survives_a_tiny_match():
    rows = [{"from_min": 0, "to_min": 5, "tempo": 15, "turnovers_per_min": 1.0,
             "pressure_volatility": 0.1}]
    analyze._classify_states(rows)
    assert rows[0]["state"] == "contested"


# -------------------------------------------------------------- key moments

def test_key_moments_are_ranked(report):
    scores = [m["importance"] for m in report["key_moments"]]
    assert scores == sorted(scores, reverse=True)


def test_every_goal_outranks_every_non_goal(report):
    moments = report["key_moments"]
    goals = [m["importance"] for m in moments if m["kind"] == "goal"]
    others = [m["importance"] for m in moments if m["kind"] != "goal"]
    if goals and others:
        assert min(goals) > max(others), "a build-up moment outranked a goal"


def test_all_goals_appear_in_key_moments(loaded):
    _, events = loaded
    names = {"HOM": "Home", "AWY": "Away"}
    scored = sum(1 for e in events if e["type"] == "shot" and e["outcome"] == "goal")
    moments = analyze.key_moments(events, names, top_n=500)
    assert sum(1 for m in moments if m["kind"] == "goal") == scored


def test_improbable_goals_rank_above_tap_ins():
    names = {"HOM": "Home"}
    base = {"type": "shot", "outcome": "goal", "team_id": "HOM",
            "player_name": "P", "clock": "10:00", "period": 1}
    moments = analyze.key_moments(
        [dict(base, xg=0.05, event_id="a"), dict(base, xg=0.85, event_id="b")],
        names,
    )
    assert moments[0]["event_id"] == "a"


def test_key_moments_carry_what_the_narrator_needs(report):
    for m in report["key_moments"]:
        assert m["clock"] and m["player"] and m["team"] and m["event_id"]
        assert m["importance"] > 0


def test_empty_input_does_not_crash():
    assert analyze.key_moments([], {}) == []
    assert analyze.player_stats([]) == []
    assert analyze.momentum_timeline([], ["HOM", "AWY"]) == []


# ------------------------------------------------------------------- report

def test_report_declares_synthetic_provenance(report):
    assert report["data_source"] == "synthetic"


def test_report_is_json_serializable(report):
    """It gets written to disk and read by the agent layer."""
    assert json.loads(json.dumps(report)) == report


def test_report_has_every_section(report):
    for key in ("match_id", "teams", "final_score", "possession_pct",
                "field_tilt_pct", "team_stats", "top_players", "momentum",
                "key_moments"):
        assert key in report, f"missing {key}"


def test_summary_prints_without_error(report, capsys):
    analyze.print_summary(report)
    out = capsys.readouterr().out
    assert "Possession %" in out
    assert "Key moments" in out
