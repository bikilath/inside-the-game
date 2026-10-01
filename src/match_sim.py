"""
Synthetic football event generator.

Produces a football-realistic event stream for a single match: passes, carries,
shots, tackles, interceptions, duels, fouls, possession changes, pressure events
and match metadata. No real match data is used or required.

Design notes
------------
* Possession is modelled as a chain of events belonging to one team until a
  turnover occurs. Chain length is drawn from a geometric-ish distribution so
  most chains are short (2-5 events) and long build-ups are rare, which matches
  the shape of real possession data.
* The pitch is 105x68 metres, origin at the bottom-left corner of the team in
  possession's own half. Coordinates are always expressed from the perspective
  of the team in possession (attacking towards x=105), which makes downstream
  metrics direction-agnostic.
* Event timestamps advance by a random interval; the clock is the single source
  of truth and every event carries both an absolute second and a period.

Usage
-----
    python match_sim.py --out match.jsonl --seed 7
    python match_sim.py --out match.jsonl --stream   # real-time-ish playback
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Iterator

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

FORMATION_433 = [
    ("GK", 1, (5.0, 34.0)),
    ("RB", 2, (25.0, 58.0)),
    ("RCB", 5, (18.0, 44.0)),
    ("LCB", 6, (18.0, 24.0)),
    ("LB", 3, (25.0, 10.0)),
    ("CDM", 4, (40.0, 34.0)),
    ("RCM", 8, (55.0, 46.0)),
    ("LCM", 10, (55.0, 22.0)),
    ("RW", 7, (78.0, 58.0)),
    ("ST", 9, (88.0, 34.0)),
    ("LW", 11, (78.0, 10.0)),
]

FIRST_NAMES = [
    "Marco", "Tomas", "Idris", "Leon", "Yuki", "Mateo", "Kwame", "Felix",
    "Diego", "Arne", "Noah", "Samir", "Oscar", "Rui", "Bilal", "Kai",
    "Emre", "Luca", "Dami", "Soren", "Ivan", "Pedro", "Omar", "Jonas",
]
LAST_NAMES = [
    "Verhoeven", "Okafor", "Lindqvist", "Marchetti", "Bakker", "Silva",
    "Novak", "Hassan", "Dubois", "Kowalski", "Arnesen", "Moretti",
    "Bergstrom", "Adeyemi", "Castillo", "Varga", "Pires", "Lindholm",
    "Osei", "Ricci", "Haugen", "Fontaine", "Markovic", "Delgado",
]

CLUB_NAMES = [
    "Ashcombe United", "Northgate FC", "Ravensmoor City", "Eastbrook Rovers",
    "Kingsmere Athletic", "Port Alder FC", "Westhaven Town", "Clayfield United",
]


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


@dataclass
class Player:
    player_id: str
    name: str
    shirt: int
    position: str
    base_x: float
    base_y: float
    team_id: str
    # rough per-player quality, drives pass completion and shot conversion
    technique: float = 0.75
    pace: float = 0.75

    def to_meta(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "shirt": self.shirt,
            "position": self.position,
            "team_id": self.team_id,
        }


@dataclass
class Team:
    team_id: str
    name: str
    short_name: str
    players: list[Player] = field(default_factory=list)
    # tactical dials that shape the event stream
    press_intensity: float = 0.5   # 0 = sit deep, 1 = aggressive high press
    directness: float = 0.5        # 0 = patient build-up, 1 = long and direct
    quality: float = 0.75

    def by_position(self, pos: str) -> Player:
        for p in self.players:
            if p.position == pos:
                return p
        return self.players[0]

    def outfield(self) -> list[Player]:
        return [p for p in self.players if p.position != "GK"]


def build_team(team_id: str, name: str, rng: random.Random) -> Team:
    short = "".join(w[0] for w in name.split()[:3]).upper()
    quality = rng.uniform(0.62, 0.88)
    team = Team(
        team_id=team_id,
        name=name,
        short_name=short,
        press_intensity=rng.uniform(0.3, 0.85),
        directness=rng.uniform(0.2, 0.75),
        quality=quality,
    )
    used: set[tuple[str, str]] = set()
    for pos, shirt, (bx, by) in FORMATION_433:
        while True:
            first = rng.choice(FIRST_NAMES)
            last = rng.choice(LAST_NAMES)
            if (first, last) not in used:
                used.add((first, last))
                break
        team.players.append(
            Player(
                player_id=f"{team_id}-{shirt:02d}",
                name=f"{first} {last}",
                shirt=shirt,
                position=pos,
                base_x=bx,
                base_y=by,
                team_id=team_id,
                technique=clamp(rng.gauss(quality, 0.07), 0.4, 0.97),
                pace=clamp(rng.gauss(quality, 0.09), 0.4, 0.97),
            )
        )
    return team


class MatchSim:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        names = self.rng.sample(CLUB_NAMES, 2)
        self.home = build_team("HOM", names[0], self.rng)
        self.away = build_team("AWY", names[1], self.rng)
        self.match_id = str(uuid.uuid4())[:8]
        self.score = {"HOM": 0, "AWY": 0}
        self.seq = 0
        self.clock = 0.0
        self.period = 1
        # momentum drifts over the match and biases who wins turnovers
        self.momentum = 0.0

    # ------------------------------------------------------------------ utils

    def other(self, team: Team) -> Team:
        return self.away if team is self.home else self.home

    def _event(self, team: Team, player: Player, etype: str, **payload) -> dict:
        self.seq += 1
        ev = {
            "event_id": f"{self.match_id}-{self.seq:05d}",
            "match_id": self.match_id,
            "sequence": self.seq,
            "period": self.period,
            "clock_s": round(self.clock, 2),
            "clock": self._clock_label(),
            "type": etype,
            "team_id": team.team_id,
            "player_id": player.player_id,
            "player_name": player.name,
            "score": dict(self.score),
        }
        ev.update(payload)
        return ev

    def _clock_label(self) -> str:
        total = int(self.clock)
        return f"{total // 60:02d}:{total % 60:02d}"

    def _advance(self, lo: float, hi: float) -> None:
        self.clock += self.rng.uniform(lo, hi)

    def _position_for(self, player: Player, jitter: float = 7.0) -> tuple[float, float]:
        x = clamp(self.rng.gauss(player.base_x, jitter), 1.0, PITCH_LENGTH - 1.0)
        y = clamp(self.rng.gauss(player.base_y, jitter), 1.0, PITCH_WIDTH - 1.0)
        return round(x, 2), round(y, 2)

    # --------------------------------------------------------------- pressure

    def _pressure(self, defending: Team, x: float) -> float:
        """How much defensive pressure is on the ball at pitch position x."""
        # pressing teams squeeze higher up; pressure also rises near own goal
        high_press = defending.press_intensity * clamp((x - 40.0) / 65.0, 0.0, 1.0)
        deep_block = (1.0 - defending.press_intensity) * clamp((x - 75.0) / 30.0, 0.0, 1.0)
        base = 0.22 + 0.55 * high_press + 0.45 * deep_block
        return clamp(base + self.rng.gauss(0.0, 0.08), 0.02, 0.98)

    # ----------------------------------------------------------------- events

    def _pass_event(self, team: Team, passer: Player, defending: Team) -> tuple[dict, Player | None, bool]:
        ox, oy = self._position_for(passer)
        pressure = self._pressure(defending, ox)

        # choose a receiver, biased forward when the team is direct
        candidates = [p for p in team.players if p is not passer]
        forward_bias = team.directness * 0.6 + self.rng.random() * 0.4
        weights = []
        for p in candidates:
            gain = p.base_x - passer.base_x
            w = math.exp((gain / 30.0) * forward_bias * 2.0)
            w *= math.exp(-abs(p.base_y - passer.base_y) / 40.0)
            weights.append(w)
        receiver = self.rng.choices(candidates, weights=weights, k=1)[0]
        dx, dy = self._position_for(receiver)

        distance = math.dist((ox, oy), (dx, dy))
        progressive = dx - ox

        # difficulty 0..1 from distance, forward progression and pressure
        difficulty = clamp(
            0.30 * clamp(distance / 45.0, 0.0, 1.0)
            + 0.30 * clamp(progressive / 35.0, 0.0, 1.0)
            + 0.40 * pressure,
            0.0,
            1.0,
        )
        p_success = clamp(0.98 - 0.42 * difficulty + 0.18 * (passer.technique - 0.75), 0.30, 0.99)
        completed = self.rng.random() < p_success

        ball_speed = clamp(self.rng.gauss(12.0 + distance * 0.22, 2.4), 4.0, 32.0)

        self._advance(1.4, 4.2)
        ev = self._event(
            team,
            passer,
            "pass",
            start={"x": ox, "y": oy},
            end={"x": dx, "y": dy},
            receiver_id=receiver.player_id if completed else None,
            receiver_name=receiver.name if completed else None,
            distance_m=round(distance, 2),
            progressive_m=round(progressive, 2),
            is_progressive=bool(progressive >= 10.0),
            difficulty=round(difficulty, 3),
            pressure=round(pressure, 3),
            ball_speed_ms=round(ball_speed, 2),
            outcome="complete" if completed else "incomplete",
        )
        return ev, (receiver if completed else None), completed

    def _carry_event(self, team: Team, player: Player, defending: Team) -> dict:
        ox, oy = self._position_for(player)
        gain = clamp(self.rng.gauss(7.0, 4.0), 0.5, 28.0)
        dx = clamp(ox + gain, 1.0, PITCH_LENGTH - 1.0)
        dy = clamp(oy + self.rng.gauss(0.0, 5.0), 1.0, PITCH_WIDTH - 1.0)
        duration = clamp(self.rng.gauss(2.6, 0.9), 0.6, 8.0)
        dist = math.dist((ox, oy), (dx, dy))
        self._advance(duration, duration + 1.5)
        return self._event(
            team,
            player,
            "carry",
            start={"x": round(ox, 2), "y": round(oy, 2)},
            end={"x": round(dx, 2), "y": round(dy, 2)},
            distance_m=round(dist, 2),
            duration_s=round(duration, 2),
            top_speed_ms=round(clamp(dist / duration * self.rng.uniform(1.0, 1.25), 1.0, 10.5), 2),
            pressure=round(self._pressure(defending, ox), 3),
        )

    def _shot_event(self, team: Team, shooter: Player, defending: Team) -> tuple[dict, bool]:
        ox, oy = self._position_for(shooter, jitter=5.0)
        ox = clamp(max(ox, 72.0), 60.0, 103.0)
        goal = (PITCH_LENGTH, PITCH_WIDTH / 2.0)
        dist = math.dist((ox, oy), goal)
        angle = abs(oy - PITCH_WIDTH / 2.0)
        pressure = self._pressure(defending, ox)

        # crude xG: decays with distance and angle, penalised by pressure
        xg = clamp(
            0.92 * math.exp(-dist / 11.0) * math.exp(-angle / 22.0) * (1.0 - 0.35 * pressure),
            0.005,
            0.92,
        )
        xg *= (0.75 + 0.5 * shooter.technique)
        xg = clamp(xg, 0.004, 0.93)

        roll = self.rng.random()
        if roll < xg:
            outcome = "goal"
        elif roll < xg + 0.30:
            outcome = "saved"
        elif roll < xg + 0.52:
            outcome = "off_target"
        else:
            outcome = "blocked"

        if outcome == "goal":
            self.score[team.team_id] += 1
            self.momentum += 0.35 if team is self.home else -0.35
            self.momentum = clamp(self.momentum, -1.0, 1.0)

        self._advance(1.0, 2.5)
        ev = self._event(
            team,
            shooter,
            "shot",
            location={"x": round(ox, 2), "y": round(oy, 2)},
            distance_m=round(dist, 2),
            xg=round(xg, 4),
            shot_speed_ms=round(clamp(self.rng.gauss(24.0, 5.0), 9.0, 38.0), 2),
            body_part=self.rng.choices(["right_foot", "left_foot", "head"], [0.55, 0.32, 0.13])[0],
            pressure=round(pressure, 3),
            outcome=outcome,
        )
        return ev, outcome == "goal"

    def _turnover_events(self, team: Team, loser: Player, defending: Team) -> list[dict]:
        """Emit the defensive action plus the possession change."""
        winner = self.rng.choice(defending.outfield())
        kind = self.rng.choices(
            ["tackle", "interception", "duel_lost", "foul"],
            [0.34, 0.30, 0.24, 0.12],
        )[0]
        ox, oy = self._position_for(loser)
        self._advance(0.8, 2.2)

        events = [
            self._event(
                defending,
                winner,
                kind,
                location={"x": ox, "y": oy},
                opponent_id=loser.player_id,
                opponent_name=loser.name,
                pressure=round(self._pressure(defending, ox), 3),
                outcome="won" if kind != "foul" else "conceded",
            )
        ]
        self._advance(0.5, 1.5)
        events.append(
            self._event(
                defending,
                winner,
                "possession_change",
                location={"x": ox, "y": oy},
                from_team=team.team_id,
                to_team=defending.team_id,
                trigger=kind,
            )
        )
        return events

    def _pressure_event(self, team: Team, defending: Team, x: float) -> dict:
        presser = self.rng.choice(defending.outfield())
        intensity = self._pressure(defending, x)
        return self._event(
            defending,
            presser,
            "pressure",
            location={"x": round(x, 2), "y": round(self.rng.uniform(5.0, 63.0), 2)},
            intensity=round(intensity, 3),
            players_committed=self.rng.randint(1, 3),
            target_team=team.team_id,
        )

    # ------------------------------------------------------------- possession

    def _possession_chain(self, team: Team) -> Iterator[dict]:
        defending = self.other(team)
        carrier = self.rng.choice(team.outfield())
        max_len = self.rng.randint(1, 11)

        for step in range(max_len):
            # occasional standalone pressure event before the action
            if self.rng.random() < 0.18:
                yield self._pressure_event(team, defending, carrier.base_x)

            # shot if we're high up the pitch and the dice say so
            in_final_third = carrier.base_x >= 70.0
            shot_chance = 0.040 if in_final_third else 0.002
            if self.rng.random() < shot_chance:
                ev, scored = self._shot_event(team, carrier, defending)
                yield ev
                return

            if self.rng.random() < 0.22:
                yield self._carry_event(team, carrier, defending)

            ev, receiver, ok = self._pass_event(team, carrier, defending)
            yield ev
            if not ok:
                for e in self._turnover_events(team, carrier, defending):
                    yield e
                return
            carrier = receiver

        # chain ran out: possession fizzles into a turnover
        for e in self._turnover_events(team, carrier, defending):
            yield e

    # ------------------------------------------------------------------- main

    def metadata(self) -> dict:
        return {
            "type": "match_metadata",
            "match_id": self.match_id,
            "competition": "Synthetic League (fictional)",
            "venue": f"{self.home.name} Stadium",
            "kickoff_utc": "2026-10-17T14:00:00Z",
            "data_source": "synthetic",
            "teams": {
                "home": {
                    "team_id": self.home.team_id,
                    "name": self.home.name,
                    "short_name": self.home.short_name,
                    "formation": "4-3-3",
                    "press_intensity": round(self.home.press_intensity, 3),
                    "directness": round(self.home.directness, 3),
                    "squad": [p.to_meta() for p in self.home.players],
                },
                "away": {
                    "team_id": self.away.team_id,
                    "name": self.away.name,
                    "short_name": self.away.short_name,
                    "formation": "4-3-3",
                    "press_intensity": round(self.away.press_intensity, 3),
                    "directness": round(self.away.directness, 3),
                    "squad": [p.to_meta() for p in self.away.players],
                },
            },
        }

    def run(self) -> Iterator[dict]:
        yield self.metadata()

        for period in (1, 2):
            self.period = period
            self.clock = 0.0 if period == 1 else 45 * 60.0
            end = (45 * 60.0) if period == 1 else (90 * 60.0)

            yield self._event(
                self.home if period == 1 else self.away,
                self.home.by_position("ST") if period == 1 else self.away.by_position("ST"),
                "kick_off",
                location={"x": 52.5, "y": 34.0},
            )

            team = self.home if period == 1 else self.away
            while self.clock < end:
                # momentum nudges who keeps the ball
                bias = 0.5 + (self.momentum * 0.18 if team is self.home else -self.momentum * 0.18)
                if self.rng.random() > clamp(bias, 0.2, 0.8):
                    team = self.other(team)
                for ev in self._possession_chain(team):
                    yield ev
                team = self.other(team)
                self.momentum = clamp(self.momentum * 0.97 + self.rng.gauss(0, 0.04), -1.0, 1.0)

            yield self._event(
                self.home,
                self.home.by_position("GK"),
                "period_end",
                location={"x": 52.5, "y": 34.0},
            )

        self.seq += 1
        yield {
            "event_id": f"{self.match_id}-{self.seq:05d}",
            "match_id": self.match_id,
            "type": "full_time",
            "clock": "90:00",
            "score": dict(self.score),
            "result": (
                "home_win" if self.score["HOM"] > self.score["AWY"]
                else "away_win" if self.score["AWY"] > self.score["HOM"]
                else "draw"
            ),
        }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic football event stream.")
    ap.add_argument("--out", default="match.jsonl", help="output JSONL path")
    ap.add_argument("--seed", type=int, default=None, help="random seed for reproducibility")
    ap.add_argument("--stream", action="store_true", help="also print events to stdout with delays")
    ap.add_argument("--speed", type=float, default=60.0, help="playback speed multiplier for --stream")
    args = ap.parse_args()

    sim = MatchSim(seed=args.seed)
    count = 0
    last_clock = 0.0

    with open(args.out, "w", encoding="utf-8") as fh:
        for ev in sim.run():
            fh.write(json.dumps(ev) + "\n")
            count += 1
            if args.stream:
                clk = ev.get("clock_s", last_clock)
                delay = max(0.0, (clk - last_clock) / max(args.speed, 0.01))
                last_clock = clk
                time.sleep(min(delay, 2.0))
                print(json.dumps(ev), flush=True)

    meta = sim.metadata()["teams"]
    print(
        f"\n{count} events -> {args.out}\n"
        f"{meta['home']['name']} {sim.score['HOM']} - {sim.score['AWY']} {meta['away']['name']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
