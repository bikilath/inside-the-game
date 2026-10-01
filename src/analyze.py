"""
Match analysis layer.

Reads the synthetic event stream produced by match_sim.py and turns it into the
football states and statistics the narrative/explain stages consume.

This is deliberately a pure, dependency-free module: every function takes events
and returns plain dicts, so it can run inside an Azure Function, a Container App
worker, or a notebook without modification.

Usage
-----
    python analyze.py ../data/match_sample.jsonl
    python analyze.py ../data/match_sample.jsonl --json report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any

Event = dict[str, Any]


# --------------------------------------------------------------------- loading

def load_events(path: str) -> tuple[Event, list[Event]]:
    """Return (metadata, events). Metadata is the first match_metadata row."""
    meta: Event = {}
    events: list[Event] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("type") == "match_metadata":
                meta = row
            else:
                events.append(row)
    return meta, events


def team_names(meta: Event) -> dict[str, str]:
    teams = meta.get("teams", {})
    return {
        teams["home"]["team_id"]: teams["home"]["name"],
        teams["away"]["team_id"]: teams["away"]["name"],
    }


# ------------------------------------------------------------------ team stats

def team_stats(events: list[Event], team_ids: list[str]) -> dict[str, dict]:
    """Core per-team totals: passing, shooting, defending, territory."""
    out = {
        t: {
            "passes": 0,
            "passes_completed": 0,
            "progressive_passes": 0,
            "pass_distance_total": 0.0,
            "pass_difficulty": [],
            "carries": 0,
            "carry_distance_total": 0.0,
            "shots": 0,
            "goals": 0,
            "xg": 0.0,
            "shots_on_target": 0,
            "tackles": 0,
            "interceptions": 0,
            "fouls": 0,
            "touches_final_third": 0,
            "pressure_events": 0,
            "pressure_applied": [],
        }
        for t in team_ids
    }

    for ev in events:
        t = ev.get("team_id")
        if t not in out:
            continue
        s = out[t]
        etype = ev["type"]

        if etype == "pass":
            s["passes"] += 1
            s["pass_distance_total"] += ev.get("distance_m", 0.0)
            s["pass_difficulty"].append(ev.get("difficulty", 0.0))
            if ev.get("outcome") == "complete":
                s["passes_completed"] += 1
                if ev.get("is_progressive"):
                    s["progressive_passes"] += 1
            if ev.get("start", {}).get("x", 0) >= 70:
                s["touches_final_third"] += 1

        elif etype == "carry":
            s["carries"] += 1
            s["carry_distance_total"] += ev.get("distance_m", 0.0)
            if ev.get("start", {}).get("x", 0) >= 70:
                s["touches_final_third"] += 1

        elif etype == "shot":
            s["shots"] += 1
            s["xg"] += ev.get("xg", 0.0)
            if ev.get("outcome") == "goal":
                s["goals"] += 1
                s["shots_on_target"] += 1
            elif ev.get("outcome") == "saved":
                s["shots_on_target"] += 1
            s["touches_final_third"] += 1

        elif etype == "tackle":
            s["tackles"] += 1
        elif etype == "interception":
            s["interceptions"] += 1
        elif etype == "foul":
            s["fouls"] += 1
        elif etype == "pressure":
            s["pressure_events"] += 1
            s["pressure_applied"].append(ev.get("intensity", 0.0))

    # derive the rates
    for t, s in out.items():
        s["pass_accuracy"] = round(100 * s["passes_completed"] / s["passes"], 1) if s["passes"] else 0.0
        s["avg_pass_distance_m"] = round(s["pass_distance_total"] / s["passes"], 2) if s["passes"] else 0.0
        s["avg_pass_difficulty"] = round(statistics.fmean(s["pass_difficulty"]), 3) if s["pass_difficulty"] else 0.0
        s["avg_pressure_applied"] = (
            round(statistics.fmean(s["pressure_applied"]), 3) if s["pressure_applied"] else 0.0
        )
        s["xg"] = round(s["xg"], 3)
        s["xg_per_shot"] = round(s["xg"] / s["shots"], 3) if s["shots"] else 0.0
        s["carry_distance_total"] = round(s["carry_distance_total"], 1)
        s["pass_distance_total"] = round(s["pass_distance_total"], 1)
        # drop the raw sample lists from the report
        del s["pass_difficulty"]
        del s["pressure_applied"]

    return out


def possession_share(events: list[Event], team_ids: list[str]) -> dict[str, float]:
    """Share of on-ball actions (passes + carries + shots) per team."""
    counts = defaultdict(int)
    for ev in events:
        if ev["type"] in ("pass", "carry", "shot"):
            counts[ev["team_id"]] += 1
    total = sum(counts.values()) or 1
    return {t: round(100 * counts[t] / total, 1) for t in team_ids}


def field_tilt(events: list[Event], team_ids: list[str]) -> dict[str, float]:
    """Share of final-third touches. The standard proxy for territorial control."""
    counts = defaultdict(int)
    for ev in events:
        if ev["type"] not in ("pass", "carry", "shot"):
            continue
        loc = ev.get("start") or ev.get("location") or {}
        if loc.get("x", 0) >= 70:
            counts[ev["team_id"]] += 1
    total = sum(counts.values()) or 1
    return {t: round(100 * counts[t] / total, 1) for t in team_ids}


# ---------------------------------------------------------------- player stats

def player_stats(events: list[Event]) -> list[dict]:
    """Per-player contribution, sorted by involvement."""
    acc: dict[str, dict] = {}

    for ev in events:
        pid = ev.get("player_id")
        if not pid:
            continue
        p = acc.setdefault(
            pid,
            {
                "player_id": pid,
                "name": ev.get("player_name"),
                "team_id": ev.get("team_id"),
                "touches": 0,
                "passes": 0,
                "passes_completed": 0,
                "progressive_passes": 0,
                "shots": 0,
                "goals": 0,
                "xg": 0.0,
                "tackles": 0,
                "interceptions": 0,
                "distance_carried_m": 0.0,
                "top_speed_ms": 0.0,
            },
        )
        etype = ev["type"]

        if etype == "pass":
            p["touches"] += 1
            p["passes"] += 1
            if ev.get("outcome") == "complete":
                p["passes_completed"] += 1
                if ev.get("is_progressive"):
                    p["progressive_passes"] += 1
        elif etype == "carry":
            p["touches"] += 1
            p["distance_carried_m"] += ev.get("distance_m", 0.0)
            p["top_speed_ms"] = max(p["top_speed_ms"], ev.get("top_speed_ms", 0.0))
        elif etype == "shot":
            p["touches"] += 1
            p["shots"] += 1
            p["xg"] += ev.get("xg", 0.0)
            if ev.get("outcome") == "goal":
                p["goals"] += 1
        elif etype == "tackle":
            p["tackles"] += 1
        elif etype == "interception":
            p["interceptions"] += 1

    rows = []
    for p in acc.values():
        p["xg"] = round(p["xg"], 3)
        p["distance_carried_m"] = round(p["distance_carried_m"], 1)
        p["pass_accuracy"] = (
            round(100 * p["passes_completed"] / p["passes"], 1) if p["passes"] else 0.0
        )
        rows.append(p)

    rows.sort(key=lambda r: r["touches"], reverse=True)
    return rows


# ------------------------------------------------------- momentum & key moments

def momentum_timeline(events: list[Event], team_ids: list[str], bucket_min: int = 5) -> list[dict]:
    """
    Bucketed match rhythm. For each window: possession split, xG created,
    shots, turnovers, and mean pressure. This is what drives a momentum chart
    and what the 'why does this matter' layer reasons over.
    """
    buckets: dict[int, dict] = {}

    for ev in events:
        clock = ev.get("clock_s")
        if clock is None:
            continue
        b = int(clock // (bucket_min * 60))
        row = buckets.setdefault(
            b,
            {
                "bucket": b,
                "from_min": b * bucket_min,
                "to_min": (b + 1) * bucket_min,
                "actions": {t: 0 for t in team_ids},
                "xg": {t: 0.0 for t in team_ids},
                "shots": {t: 0 for t in team_ids},
                "turnovers": 0,
                "pressure": [],
            },
        )
        t = ev.get("team_id")
        etype = ev["type"]

        if etype in ("pass", "carry", "shot") and t in row["actions"]:
            row["actions"][t] += 1
        if etype == "shot" and t in row["xg"]:
            row["xg"][t] += ev.get("xg", 0.0)
            row["shots"][t] += 1
        if etype == "possession_change":
            row["turnovers"] += 1
        if "pressure" in ev:
            row["pressure"].append(ev["pressure"])
        elif etype == "pressure":
            row["pressure"].append(ev.get("intensity", 0.0))

    out = []
    for b in sorted(buckets):
        row = buckets[b]
        total_actions = sum(row["actions"].values()) or 1
        row["possession_pct"] = {
            t: round(100 * row["actions"][t] / total_actions, 1) for t in team_ids
        }
        row["xg"] = {t: round(v, 3) for t, v in row["xg"].items()}
        row["avg_pressure"] = round(statistics.fmean(row["pressure"]), 3) if row["pressure"] else 0.0
        # control vs chaos: low pressure variance + few turnovers = controlled
        row["pressure_volatility"] = (
            round(statistics.pstdev(row["pressure"]), 3) if len(row["pressure"]) > 1 else 0.0
        )
        row["tempo"] = round(total_actions / bucket_min, 2)  # actions per minute
        row["turnovers_per_min"] = round(
            row["turnovers"] / max(row["to_min"] - row["from_min"], 1), 2
        )
        del row["pressure"]
        out.append(row)

    _classify_states(out)
    return out


def _classify_states(rows: list[dict]) -> None:
    """
    Label each window controlled / contested / chaotic, relative to this match.

    Absolute thresholds don't transfer between matches (a high-press game has a
    higher baseline turnover rate throughout), so the state is scored against
    the match's own median: a window is chaotic when both turnover rate and
    pressure volatility run hot for *this* game, controlled when both run cold.
    """
    full = [r for r in rows if (r["to_min"] - r["from_min"]) >= 5 and r["tempo"] > 5]
    if len(full) < 3:
        for r in rows:
            r["state"] = "contested"
        return

    med_to = statistics.median(r["turnovers_per_min"] for r in full)
    med_vol = statistics.median(r["pressure_volatility"] for r in full)

    for r in rows:
        if r not in full:
            r["state"] = "contested"
            continue
        hot = (r["turnovers_per_min"] > med_to) + (r["pressure_volatility"] > med_vol)
        cold = (r["turnovers_per_min"] < med_to) + (r["pressure_volatility"] < med_vol)
        if hot == 2:
            r["state"] = "chaotic"
        elif cold == 2:
            r["state"] = "controlled"
        else:
            r["state"] = "contested"


def key_moments(events: list[Event], names: dict[str, str], top_n: int = 12) -> list[dict]:
    """
    Rank events by how much they mattered, so the narrative layer has something
    to explain rather than a flat list of everything that happened.
    """
    moments = []
    for ev in events:
        etype = ev["type"]
        score = 0.0
        kind = None

        if etype == "shot":
            xg = ev.get("xg", 0.0)
            if ev.get("outcome") == "goal":
                # a goal is categorically more important than any build-up play;
                # low-xG goals score highest of all (the improbable ones)
                score = 300 + (1.0 - xg) * 60
                kind = "goal"
            else:
                score = xg * 120
                kind = f"shot_{ev.get('outcome')}"
        elif etype == "pass" and ev.get("outcome") == "complete":
            # a hard, progressive pass under pressure is a real moment
            if ev.get("difficulty", 0) >= 0.55 and ev.get("progressive_m", 0) >= 18:
                score = 25 + ev["difficulty"] * 25 + ev["progressive_m"] * 0.4
                kind = "line_breaking_pass"
        elif etype in ("tackle", "interception"):
            x = (ev.get("location") or {}).get("x", 0)
            if x >= 65:  # won high up the pitch
                score = 20 + (x - 65) * 0.8
                kind = f"high_{etype}"

        if kind and score > 0:
            moments.append(
                {
                    "clock": ev.get("clock"),
                    "period": ev.get("period"),
                    "kind": kind,
                    "team": names.get(ev.get("team_id"), ev.get("team_id")),
                    "player": ev.get("player_name"),
                    "importance": round(score, 1),
                    "xg": ev.get("xg"),
                    "difficulty": ev.get("difficulty"),
                    "event_id": ev.get("event_id"),
                }
            )

    moments.sort(key=lambda m: m["importance"], reverse=True)
    return moments[:top_n]


# ------------------------------------------------------------------- reporting

def build_report(path: str) -> dict:
    meta, events = load_events(path)
    names = team_names(meta)
    tids = list(names)

    report = {
        "match_id": meta.get("match_id"),
        "teams": names,
        "data_source": meta.get("data_source", "synthetic"),
        "final_score": next(
            (e["score"] for e in reversed(events) if e.get("type") == "full_time"), {}
        ),
        "event_count": len(events),
        "possession_pct": possession_share(events, tids),
        "field_tilt_pct": field_tilt(events, tids),
        "team_stats": team_stats(events, tids),
        "top_players": player_stats(events)[:10],
        "momentum": momentum_timeline(events, tids),
        "key_moments": key_moments(events, names),
    }
    return report


def print_summary(r: dict) -> None:
    names = r["teams"]
    tids = list(names)
    a, b = tids[0], tids[1]
    score = r["final_score"]

    print(f"\n{names[a]} {score.get(a, 0)} - {score.get(b, 0)} {names[b]}")
    print(f"{r['event_count']} events  |  match {r['match_id']}  |  source: {r['data_source']}\n")

    rows = [
        ("Possession %", r["possession_pct"][a], r["possession_pct"][b]),
        ("Field tilt %", r["field_tilt_pct"][a], r["field_tilt_pct"][b]),
        ("Shots", r["team_stats"][a]["shots"], r["team_stats"][b]["shots"]),
        ("xG", r["team_stats"][a]["xg"], r["team_stats"][b]["xg"]),
        ("Passes", r["team_stats"][a]["passes"], r["team_stats"][b]["passes"]),
        ("Pass accuracy %", r["team_stats"][a]["pass_accuracy"], r["team_stats"][b]["pass_accuracy"]),
        ("Progressive passes", r["team_stats"][a]["progressive_passes"], r["team_stats"][b]["progressive_passes"]),
        ("Final-third touches", r["team_stats"][a]["touches_final_third"], r["team_stats"][b]["touches_final_third"]),
        ("Tackles", r["team_stats"][a]["tackles"], r["team_stats"][b]["tackles"]),
        ("Interceptions", r["team_stats"][a]["interceptions"], r["team_stats"][b]["interceptions"]),
        ("Avg pressure applied", r["team_stats"][a]["avg_pressure_applied"], r["team_stats"][b]["avg_pressure_applied"]),
    ]
    w = max(len(x[0]) for x in rows)
    print(f"{'':<{w}}  {names[a][:18]:>18}  {names[b][:18]:>18}")
    for label, va, vb in rows:
        print(f"{label:<{w}}  {str(va):>18}  {str(vb):>18}")

    print("\nMatch rhythm")
    for m in r["momentum"]:
        bar_a = int(m["possession_pct"][a] / 5)
        print(
            f"  {m['from_min']:>2}-{m['to_min']:<2}min  "
            f"{m['possession_pct'][a]:>5.1f}% {'#' * bar_a:<20} "
            f"tempo {m['tempo']:>5.2f}  {m['state']}"
        )

    print("\nKey moments")
    for m in r["key_moments"]:
        extra = f" xG {m['xg']}" if m.get("xg") else ""
        print(f"  {m['clock']}  [{m['importance']:>6.1f}]  {m['kind']:<22} {m['player']} ({m['team']}){extra}")

    print("\nMost involved players")
    for p in r["top_players"][:6]:
        print(
            f"  {p['name']:<22} {names.get(p['team_id'], '')[:14]:<14} "
            f"touches {p['touches']:>3}  pass% {p['pass_accuracy']:>5.1f}  "
            f"prog {p['progressive_passes']:>2}  xG {p['xg']:.2f}"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze a synthetic match event stream.")
    ap.add_argument("events", help="path to the .jsonl event file")
    ap.add_argument("--json", help="also write the full report to this JSON path")
    args = ap.parse_args()

    report = build_report(args.events)
    print_summary(report)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"Full report written to {args.json}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
