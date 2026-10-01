"""
Tests for the synthetic event generator.

Two kinds of test here, and the second kind is the one that matters:

1. Structural tests  - the stream is well-formed (ids unique, clock monotonic,
   every event carries the fields downstream code reads).
2. Distribution tests - the football is plausible. A generator that runs but
   produces 176 shots a game is worse than useless: everything built on top of
   it reasons confidently about a sport that doesn't exist. These assert the
   output lands inside real-world ranges, across several seeds.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from match_sim import PITCH_LENGTH, PITCH_WIDTH, MatchSim, clamp  # noqa: E402

SEEDS = [7, 11, 23, 42, 101]


def run(seed: int) -> tuple[dict, list[dict]]:
    """Generate one match; return (metadata, events-without-metadata)."""
    rows = list(MatchSim(seed=seed).run())
    meta = rows[0]
    return meta, rows[1:]


@pytest.fixture(scope="module")
def matches() -> dict[int, tuple[dict, list[dict]]]:
    return {s: run(s) for s in SEEDS}


# --------------------------------------------------------------- determinism

def test_same_seed_gives_identical_stream():
    """The demo depends on this: rehearse against a match you know."""
    a = [e for e in MatchSim(seed=7).run()]
    b = [e for e in MatchSim(seed=7).run()]
    assert a == b


def test_different_seeds_give_different_matches():
    a = list(MatchSim(seed=7).run())
    b = list(MatchSim(seed=8).run())
    assert a != b


# --------------------------------------------------------------- structural

def test_event_ids_unique(matches):
    for seed, (_, events) in matches.items():
        ids = [e["event_id"] for e in events]
        assert len(ids) == len(set(ids)), f"duplicate event_id at seed {seed}"


def test_sequence_strictly_increasing(matches):
    for seed, (_, events) in matches.items():
        seqs = [e["sequence"] for e in events if "sequence" in e]
        assert seqs == sorted(seqs), f"sequence out of order at seed {seed}"
        assert len(seqs) == len(set(seqs))


def test_clock_never_goes_backwards_within_a_period(matches):
    for seed, (_, events) in matches.items():
        last = {1: -1.0, 2: -1.0}
        for e in events:
            p, c = e.get("period"), e.get("clock_s")
            if p in last and c is not None:
                assert c >= last[p], f"clock went backwards at seed {seed}"
                last[p] = c


def test_both_periods_present_and_full_time_last(matches):
    for seed, (_, events) in matches.items():
        periods = {e.get("period") for e in events if e.get("period")}
        assert periods == {1, 2}, f"missing a period at seed {seed}"
        assert events[-1]["type"] == "full_time"


def test_coordinates_inside_the_pitch(matches):
    for seed, (_, events) in matches.items():
        for e in events:
            for key in ("start", "end", "location"):
                loc = e.get(key)
                if not loc:
                    continue
                assert 0 <= loc["x"] <= PITCH_LENGTH, f"x off pitch at seed {seed}"
                assert 0 <= loc["y"] <= PITCH_WIDTH, f"y off pitch at seed {seed}"


def test_every_event_has_the_fields_analysis_reads(matches):
    required = {"event_id", "match_id", "type", "clock", "score"}
    for seed, (_, events) in matches.items():
        for e in events:
            missing = required - e.keys()
            assert not missing, f"{e['type']} missing {missing} at seed {seed}"


def test_passes_are_internally_consistent(matches):
    for seed, (_, events) in matches.items():
        for e in (x for x in events if x["type"] == "pass"):
            expected = math.dist(
                (e["start"]["x"], e["start"]["y"]), (e["end"]["x"], e["end"]["y"])
            )
            assert e["distance_m"] == pytest.approx(expected, abs=0.02)
            assert e["progressive_m"] == pytest.approx(
                e["end"]["x"] - e["start"]["x"], abs=0.02
            )
            assert 0.0 <= e["difficulty"] <= 1.0
            assert 0.0 <= e["pressure"] <= 1.0
            # a completed pass names its receiver; an incomplete one does not
            if e["outcome"] == "complete":
                assert e["receiver_id"]
            else:
                assert e["receiver_id"] is None


def test_score_only_ever_increases_and_matches_goals(matches):
    for seed, (_, events) in matches.items():
        goals = {"HOM": 0, "AWY": 0}
        prev = {"HOM": 0, "AWY": 0}
        for e in events:
            if e.get("type") == "shot" and e["outcome"] == "goal":
                goals[e["team_id"]] += 1
            s = e.get("score")
            if s:
                for t in ("HOM", "AWY"):
                    assert s[t] >= prev[t], f"score decreased at seed {seed}"
                prev = s
        final = events[-1]["score"]
        assert final == goals, f"final score != goals scored at seed {seed}"


def test_possession_change_names_both_teams(matches):
    for seed, (_, events) in matches.items():
        for e in (x for x in events if x["type"] == "possession_change"):
            assert e["from_team"] != e["to_team"]


def test_squads_are_complete_and_distinct(matches):
    for seed, (meta, _) in matches.items():
        for side in ("home", "away"):
            squad = meta["teams"][side]["squad"]
            assert len(squad) == 11
            assert len({p["player_id"] for p in squad}) == 11
            assert len({p["name"] for p in squad}) == 11
            assert sum(1 for p in squad if p["position"] == "GK") == 1
        home = {p["name"] for p in meta["teams"]["home"]["squad"]}
        away = {p["name"] for p in meta["teams"]["away"]["squad"]}
        assert not (home & away), f"a player is on both teams at seed {seed}"


def test_data_is_declared_synthetic(matches):
    """The hackathon forbids real match data. Keep it unambiguous."""
    for _, (meta, _) in matches.items():
        assert meta["data_source"] == "synthetic"


# ------------------------------------------------------------- distribution

def stats(events: list[dict]) -> dict:
    passes = [e for e in events if e["type"] == "pass"]
    shots = [e for e in events if e["type"] == "shot"]
    return {
        "events": len(events),
        "passes": len(passes),
        "pass_accuracy": sum(1 for p in passes if p["outcome"] == "complete") / len(passes),
        "shots": len(shots),
        "goals": sum(1 for s in shots if s["outcome"] == "goal"),
        "xg": sum(s["xg"] for s in shots),
    }


@pytest.mark.parametrize("seed", SEEDS)
def test_event_volume_is_realistic(seed, matches):
    _, events = matches[seed]
    n = stats(events)["events"]
    assert 1500 <= n <= 3500, f"{n} events is not a football match"


@pytest.mark.parametrize("seed", SEEDS)
def test_pass_accuracy_is_realistic(seed, matches):
    """Real matches sit around 75-85%. The first build of this came out at 64%."""
    _, events = matches[seed]
    acc = stats(events)["pass_accuracy"]
    assert 0.68 <= acc <= 0.90, f"pass accuracy {acc:.1%} is not plausible"


@pytest.mark.parametrize("seed", SEEDS)
def test_shot_count_is_realistic(seed, matches):
    """Both teams combined: roughly 15-35. The first build produced 176."""
    _, events = matches[seed]
    n = stats(events)["shots"]
    assert 10 <= n <= 40, f"{n} shots in a match is not plausible"


@pytest.mark.parametrize("seed", SEEDS)
def test_total_xg_is_realistic(seed, matches):
    _, events = matches[seed]
    xg = stats(events)["xg"]
    assert 0.4 <= xg <= 5.0, f"total xG of {xg:.2f} is not plausible"


@pytest.mark.parametrize("seed", SEEDS)
def test_scoreline_is_realistic(seed, matches):
    _, events = matches[seed]
    assert stats(events)["goals"] <= 8


def test_goals_track_xg_across_seeds(matches):
    """
    Over several matches, goals should land in the neighbourhood of total xG.
    Badly wrong if conversion is miscalibrated, even when each match looks fine.
    """
    total_xg = sum(stats(e)["xg"] for _, e in matches.values())
    total_goals = sum(stats(e)["goals"] for _, e in matches.values())
    assert 0.4 * total_xg <= total_goals <= 2.5 * total_xg, (
        f"{total_goals} goals from {total_xg:.1f} xG is miscalibrated"
    )


def test_xg_values_are_probabilities(matches):
    for _, (_, events) in matches.items():
        for s in (e for e in events if e["type"] == "shot"):
            assert 0.0 < s["xg"] < 1.0


def test_shots_come_from_attacking_areas(matches):
    for seed, (_, events) in matches.items():
        for s in (e for e in events if e["type"] == "shot"):
            assert s["location"]["x"] >= 55, f"shot from own half at seed {seed}"


def test_closer_shots_have_higher_xg(matches):
    """xG must fall with distance, or the explain layer reasons on nonsense."""
    _, events = matches[7]
    shots = [e for e in events if e["type"] == "shot"]
    near = [s["xg"] for s in shots if s["distance_m"] < 12]
    far = [s["xg"] for s in shots if s["distance_m"] > 25]
    if near and far:
        assert sum(near) / len(near) > sum(far) / len(far)


def test_speeds_are_humanly_possible(matches):
    for seed, (_, events) in matches.items():
        for e in events:
            if e["type"] == "carry":
                assert 0 < e["top_speed_ms"] <= 12.0, f"sprint too fast at seed {seed}"
            if e["type"] == "shot":
                assert 5.0 <= e["shot_speed_ms"] <= 50.0
            if e["type"] == "pass":
                assert 0 < e["ball_speed_ms"] <= 40.0


def test_both_teams_get_meaningful_possession(matches):
    """Neither side should be shut out of the ball entirely."""
    for seed, (_, events) in matches.items():
        onball = [e for e in events if e["type"] in ("pass", "carry", "shot")]
        home = sum(1 for e in onball if e["team_id"] == "HOM") / len(onball)
        assert 0.3 <= home <= 0.7, f"possession {home:.0%}/{1-home:.0%} at seed {seed}"


def test_pressure_events_are_well_formed(matches):
    for _, (_, events) in matches.items():
        for e in (x for x in events if x["type"] == "pressure"):
            assert 0.0 <= e["intensity"] <= 1.0
            assert 1 <= e["players_committed"] <= 3


# ------------------------------------------------------------------ helpers

def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
