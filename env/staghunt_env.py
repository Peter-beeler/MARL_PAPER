"""
Stag Hunt environment — Coordination and Trust.

Legend:
- '*' = land (all cells)
- 's' = stag (high reward, requires 2+ agents to catch)
- 'h' = hare (low reward, requires 1 agent to catch)
- '1'..'9' = agents (rendered over prey over terrain)

Scoring:
- Catching a stag: stag_reward per participating agent (default 5.0)
- Catching a hare: hare_reward per participating agent (default 1.0)

Stags and hares are mobile prey that wander the grid. To catch prey,
agents must use the 'hunt' action while adjacent (Manhattan distance <= 1).
- Stag: requires 2+ agents adjacent and hunting on the same timestep.
- Hare: requires 1+ agent adjacent and hunting.

Dilemma: Hunting a stag is the social optimum but risky — if your partner
defects to chase a hare, you waste your turn and get nothing.

This file implements StagHuntEnv with a minimal API:
- reset() -> observation
- step(actions: dict[int, str|int]) -> (obs, rewards, done, info)
- render() -> str
- get_state() -> dict
- set_state(state) -> None

You can run this file directly to play a quick random-policy demo.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------
# Config
# ---------------------------

@dataclass
class Config:
    width: int = 16
    height: int = 16
    n_agents: int = 4  # supports up to 9 for rendering with digits

    # Episode control
    max_steps: int = 300
    seed: Optional[int] = 42

    # Prey counts
    n_stags: int = 2
    n_hares: int = 4

    # Rewards
    stag_reward: float = 5.0   # per agent that participates in a stag catch
    hare_reward: float = 1.0   # per agent that participates in a hare catch

    # Prey movement probabilities (per step)
    stag_move_prob: float = 0.2   # stags move slowly
    hare_move_prob: float = 0.5   # hares are quicker

    # Respawn
    respawn_delay: int = 10  # steps before caught prey reappears

Action = int  # 0: stay, 1: hunt, 2: up, 3: down, 4: left, 5: right


class StagHuntEnv:
    """Stag Hunt grid environment.

    Layers:
      - terrain[y][x]: '*' (all land)
      - stags: dict of stag_id -> (x, y) or None (if caught)
      - hares: dict of hare_id -> (x, y) or None (if caught)
      - agents: dict of agent_id -> (x, y)
    Rendering overlays agents > prey > terrain.
    """

    # Action vocabulary (both string and int supported)
    ACTIONS = {
        0: "stay", 1: "hunt", 2: "up", 3: "down", 4: "left", 5: "right",
        "stay": "stay", "hunt": "hunt",
        "up": "up", "down": "down", "left": "left", "right": "right"
    }

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.rng = random.Random(self.cfg.seed)

        self.width = self.cfg.width
        self.height = self.cfg.height

        # Layers
        self.terrain: List[List[str]] = []  # all '*'
        self.agents: Dict[int, Tuple[int, int]] = {}
        self.prev_agents: Dict[int, Tuple[int, int]] = {}

        # Prey: id -> position (None if caught and respawning)
        self.stags: Dict[int, Optional[Tuple[int, int]]] = {}
        self.hares: Dict[int, Optional[Tuple[int, int]]] = {}

        # Respawn timers: id -> steps remaining until respawn
        self.stag_respawn: Dict[int, int] = {}
        self.hare_respawn: Dict[int, int] = {}

        # Episode state
        self.step_count = 0
        self.scores: Dict[int, float] = {}
        self.total_stag_catches = 0
        self.total_hare_catches = 0

        self.reset()

    # ---------------------------
    # Core API
    # ---------------------------
    def reset(self):
        """Reset the environment and return the initial observation."""
        self.step_count = 0
        self.scores = {i: 0.0 for i in self._agent_ids}
        self.total_stag_catches = 0
        self.total_hare_catches = 0

        # All land terrain
        self.terrain = [["*" for _ in range(self.width)] for _ in range(self.height)]

        # Place entities on random cells (no overlap between any entities)
        all_cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        self.rng.shuffle(all_cells)

        idx = 0
        total_entities = self.cfg.n_agents + self.cfg.n_stags + self.cfg.n_hares
        if len(all_cells) < total_entities:
            raise ValueError("Grid too small for all entities.")

        # Place agents
        self.agents = {}
        for i in range(1, self.cfg.n_agents + 1):
            self.agents[i] = all_cells[idx]
            idx += 1

        # Place stags
        self.stags = {}
        self.stag_respawn = {}
        for sid in range(1, self.cfg.n_stags + 1):
            self.stags[sid] = all_cells[idx]
            self.stag_respawn[sid] = 0
            idx += 1

        # Place hares
        self.hares = {}
        self.hare_respawn = {}
        for hid in range(1, self.cfg.n_hares + 1):
            self.hares[hid] = all_cells[idx]
            self.hare_respawn[hid] = 0
            idx += 1

        self.prev_agents = {i: pos for i, pos in self.agents.items()}
        return self._observation()

    def step(self, actions: Dict[int, Action | str]):
        """Advance the environment by one step.

        actions: mapping of agent_id -> action (int or str). Missing IDs default to stay.

        Returns: (observation, rewards, done, info)
        """
        self.step_count += 1
        rewards = {i: 0.0 for i in self._agent_ids}
        step_stag_catches = 0
        step_hare_catches = 0

        # Compute desired moves
        desired: Dict[int, Tuple[int, int]] = {}
        chosen_action: Dict[int, str] = {}
        for i in self._agent_ids:
            ax, ay = self.agents[i]
            a_raw = actions.get(i, 0)
            a_name = self.ACTIONS.get(
                a_raw if a_raw in self.ACTIONS else
                (a_raw.lower() if isinstance(a_raw, str) else a_raw), "stay")
            chosen_action[i] = a_name

            if a_name in ("stay", "hunt"):
                desired[i] = (ax, ay)
            elif a_name == "up":
                desired[i] = (ax, max(0, ay - 1))
            elif a_name == "down":
                desired[i] = (ax, min(self.height - 1, ay + 1))
            elif a_name == "left":
                desired[i] = (max(0, ax - 1), ay)
            elif a_name == "right":
                desired[i] = (min(self.width - 1, ax + 1), ay)
            else:
                desired[i] = (ax, ay)

        # Resolve collisions: agents moving to same cell all stay
        target_counts: Dict[Tuple[int, int], int] = {}
        for pos in desired.values():
            target_counts[pos] = target_counts.get(pos, 0) + 1

        new_positions: Dict[int, Tuple[int, int]] = {}
        for i in self._agent_ids:
            pos = desired[i]
            if target_counts[pos] > 1 and pos != self.agents[i]:
                new_positions[i] = self.agents[i]
                continue
            if pos in new_positions.values():
                new_positions[i] = self.agents[i]
                continue
            new_positions[i] = pos

        self.prev_agents = {i: pos for i, pos in self.agents.items()}
        self.agents = new_positions

        # Collect hunting agents
        hunting_agents = {i for i in self._agent_ids if chosen_action.get(i) == "hunt"}

        # Process stag hunts (need 2+ adjacent hunters)
        for sid, spos in list(self.stags.items()):
            if spos is None:
                continue
            adjacent_hunters = [i for i in hunting_agents
                                if self._is_adjacent(self.agents[i], spos)]
            if len(adjacent_hunters) >= 2:
                for i in adjacent_hunters:
                    rewards[i] += self.cfg.stag_reward
                    self.scores[i] += self.cfg.stag_reward
                self.stags[sid] = None
                self.stag_respawn[sid] = self.cfg.respawn_delay
                step_stag_catches += 1

        # Process hare hunts (need 1+ adjacent hunter)
        for hid, hpos in list(self.hares.items()):
            if hpos is None:
                continue
            adjacent_hunters = [i for i in hunting_agents
                                if self._is_adjacent(self.agents[i], hpos)]
            if len(adjacent_hunters) >= 1:
                for i in adjacent_hunters:
                    rewards[i] += self.cfg.hare_reward
                    self.scores[i] += self.cfg.hare_reward
                self.hares[hid] = None
                self.hare_respawn[hid] = self.cfg.respawn_delay
                step_hare_catches += 1

        self.total_stag_catches += step_stag_catches
        self.total_hare_catches += step_hare_catches

        # Move surviving prey
        self._move_prey()

        # Respawn caught prey
        self._respawn_prey()

        done = self.step_count >= self.cfg.max_steps
        obs = self._observation()
        info = {
            "step": self.step_count,
            "scores": dict(self.scores),
            "stags_alive": sum(1 for p in self.stags.values() if p is not None),
            "hares_alive": sum(1 for p in self.hares.values() if p is not None),
            "step_stag_catches": step_stag_catches,
            "step_hare_catches": step_hare_catches,
            "total_stag_catches": self.total_stag_catches,
            "total_hare_catches": self.total_hare_catches,
        }
        return obs, rewards, done, info

    def render(self) -> str:
        """Return a human-readable string of the current grid."""
        grid_chars = [[self.terrain[y][x] for x in range(self.width)]
                      for y in range(self.height)]
        # Overlay stags
        for sid, spos in self.stags.items():
            if spos is not None:
                grid_chars[spos[1]][spos[0]] = 's'
        # Overlay hares
        for hid, hpos in self.hares.items():
            if hpos is not None:
                grid_chars[hpos[1]][hpos[0]] = 'h'
        # Overlay agents (on top of everything)
        for i, (x, y) in self.agents.items():
            grid_chars[y][x] = str(i % 10)
        return "\n".join("".join(row) for row in grid_chars)

    def get_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current environment state."""
        return {
            "terrain": [row[:] for row in self.terrain],
            "agents": {aid: (pos[0], pos[1]) for aid, pos in self.agents.items()},
            "prev_agents": {aid: (pos[0], pos[1]) for aid, pos in self.prev_agents.items()},
            "stags": {sid: (pos[0], pos[1]) if pos is not None else None
                      for sid, pos in self.stags.items()},
            "hares": {hid: (pos[0], pos[1]) if pos is not None else None
                      for hid, pos in self.hares.items()},
            "stag_respawn": dict(self.stag_respawn),
            "hare_respawn": dict(self.hare_respawn),
            "scores": dict(self.scores),
            "step_count": self.step_count,
            "total_stag_catches": self.total_stag_catches,
            "total_hare_catches": self.total_hare_catches,
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by get_state()."""
        self.terrain = [row[:] for row in state["terrain"]]
        self.agents = {int(aid): (pos[0], pos[1])
                       for aid, pos in state["agents"].items()}
        if "prev_agents" in state:
            self.prev_agents = {int(aid): (pos[0], pos[1])
                                for aid, pos in state["prev_agents"].items()}
        else:
            self.prev_agents = {int(aid): (pos[0], pos[1])
                                for aid, pos in state["agents"].items()}
        self.stags = {int(sid): ((pos[0], pos[1]) if pos is not None else None)
                      for sid, pos in state["stags"].items()}
        self.hares = {int(hid): ((pos[0], pos[1]) if pos is not None else None)
                      for hid, pos in state["hares"].items()}
        self.stag_respawn = {int(sid): int(t)
                             for sid, t in state["stag_respawn"].items()}
        self.hare_respawn = {int(hid): int(t)
                             for hid, t in state["hare_respawn"].items()}
        self.scores = {int(aid): float(score)
                       for aid, score in state["scores"].items()}
        self.step_count = int(state["step_count"])
        self.total_stag_catches = int(state["total_stag_catches"])
        self.total_hare_catches = int(state["total_hare_catches"])
        self.rng.setstate(state["rng_state"])

    # ---------------------------
    # Helpers
    # ---------------------------
    @property
    def _agent_ids(self) -> List[int]:
        return list(range(1, self.cfg.n_agents + 1))

    @staticmethod
    def _is_adjacent(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """Check if two positions are adjacent (Manhattan distance <= 1)."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) <= 1

    def _move_prey(self):
        """Move surviving stags and hares randomly."""
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

        for sid, spos in list(self.stags.items()):
            if spos is None:
                continue
            if self.rng.random() < self.cfg.stag_move_prob:
                dx, dy = self.rng.choice(directions)
                nx, ny = spos[0] + dx, spos[1] + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.stags[sid] = (nx, ny)

        for hid, hpos in list(self.hares.items()):
            if hpos is None:
                continue
            if self.rng.random() < self.cfg.hare_move_prob:
                dx, dy = self.rng.choice(directions)
                nx, ny = hpos[0] + dx, hpos[1] + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    self.hares[hid] = (nx, ny)

    def _respawn_prey(self):
        """Decrement respawn timers and respawn prey when ready."""
        # Collect occupied positions to avoid respawning on them
        occupied = set(self.agents.values())
        for pos in self.stags.values():
            if pos is not None:
                occupied.add(pos)
        for pos in self.hares.values():
            if pos is not None:
                occupied.add(pos)

        for sid in list(self.stag_respawn.keys()):
            if self.stags[sid] is not None:
                continue
            self.stag_respawn[sid] -= 1
            if self.stag_respawn[sid] <= 0:
                pos = self._random_empty_cell(occupied)
                if pos is not None:
                    self.stags[sid] = pos
                    occupied.add(pos)
                    self.stag_respawn[sid] = 0

        for hid in list(self.hare_respawn.keys()):
            if self.hares[hid] is not None:
                continue
            self.hare_respawn[hid] -= 1
            if self.hare_respawn[hid] <= 0:
                pos = self._random_empty_cell(occupied)
                if pos is not None:
                    self.hares[hid] = pos
                    occupied.add(pos)
                    self.hare_respawn[hid] = 0

    def _random_empty_cell(self, occupied: set) -> Optional[Tuple[int, int]]:
        """Find a random cell not in the occupied set."""
        candidates = [(x, y) for y in range(self.height) for x in range(self.width)
                       if (x, y) not in occupied]
        if not candidates:
            return None
        return self.rng.choice(candidates)

    def get_agent_movement(self, agent_id: int) -> Tuple[str, Tuple[int, int]]:
        """Get the movement direction and previous position for an agent.

        Returns:
            Tuple of (direction_string, prev_position_display_coords)
            direction_string is one of: "UP", "DOWN", "LEFT", "RIGHT", "STAYED"
            prev_position uses display coordinates (y=0 at bottom)
        """
        if agent_id not in self.agents or agent_id not in self.prev_agents:
            return ("STAYED", (0, 0))

        curr_x, curr_y = self.agents[agent_id]
        prev_x, prev_y = self.prev_agents[agent_id]

        dx = curr_x - prev_x
        dy = curr_y - prev_y

        if dx == 0 and dy == 0:
            direction = "STAYED"
        elif dy < 0:
            direction = "UP"
        elif dy > 0:
            direction = "DOWN"
        elif dx < 0:
            direction = "LEFT"
        else:
            direction = "RIGHT"

        prev_y_display = (self.height - 1) - prev_y
        return (direction, (prev_x, prev_y_display))

    def _local_render(self, agent_id: int) -> str:
        """Render a 5x5 local observation for the given agent."""
        ax, ay = self.agents[agent_id]
        half_w = 2
        half_h = 2
        y0 = max(0, ay - half_h)
        y1 = min(self.height - 1, ay + half_h)
        x0 = max(0, ax - half_w)
        x1 = min(self.width - 1, ax + half_w)

        grid_chars = [[self.terrain[y][x] for x in range(x0, x1 + 1)]
                      for y in range(y0, y1 + 1)]
        # Overlay stags
        for sid, spos in self.stags.items():
            if spos is not None:
                sx, sy = spos
                if x0 <= sx <= x1 and y0 <= sy <= y1:
                    grid_chars[sy - y0][sx - x0] = 's'
        # Overlay hares
        for hid, hpos in self.hares.items():
            if hpos is not None:
                hx, hy = hpos
                if x0 <= hx <= x1 and y0 <= hy <= y1:
                    grid_chars[hy - y0][hx - x0] = 'h'
        # Overlay agents
        for aid, (x, y) in self.agents.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                grid_chars[y - y0][x - x0] = str(aid % 10)
        return "\n".join("".join(row) for row in grid_chars)

    def _observation(self):
        """Return per-agent local observations (5x5 window)."""
        return {i: self._local_render(i) for i in self._agent_ids}

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def find_nearest_prey(self, agent_id: int, prey_type: str = "stag",
                          n: int = 1) -> List[Tuple[int, Tuple[int, int]]]:
        """Find N nearest alive prey of a given type.

        Args:
            agent_id: Agent ID
            prey_type: "stag" or "hare"
            n: Number of nearest prey to return

        Returns:
            List of (prey_id, (x, y)) sorted by distance
        """
        agent_pos = self.agents[agent_id]
        prey_dict = self.stags if prey_type == "stag" else self.hares

        prey_with_dist = []
        for pid, ppos in prey_dict.items():
            if ppos is None:
                continue
            dist = self._calculate_distance(agent_pos, ppos)
            prey_with_dist.append((pid, ppos, dist))

        prey_with_dist.sort(key=lambda x: x[2])
        return [(pid, ppos) for pid, ppos, _ in prey_with_dist[:n]]


# ---------------------------
# Quick demo
# ---------------------------
def _demo():
    cfg = Config(width=12, height=12, n_agents=4, n_stags=2, n_hares=3,
                 max_steps=30, stag_move_prob=0.3, hare_move_prob=0.5)
    env = StagHuntEnv(cfg)
    print("=== Stag Hunt Demo ===")
    print("Initial grid:")
    print(env.render())
    print()

    while True:
        actions = {i: env.rng.choice([0, 1, 2, 3, 4, 5]) for i in env._agent_ids}
        obs, rewards, done, info = env.step(actions)
        if env.step_count % 5 == 0 or done:
            print(f"Step {info['step']}: stags_alive={info['stags_alive']} "
                  f"hares_alive={info['hares_alive']} "
                  f"total_stag_catches={info['total_stag_catches']} "
                  f"total_hare_catches={info['total_hare_catches']}")
            print(f"scores={info['scores']}")
            print(env.render())
            print()
        if done:
            break

    print("Final scores:", info['scores'])
    print(f"Total: stag catches={info['total_stag_catches']}, "
          f"hare catches={info['total_hare_catches']}")


if __name__ == "__main__":
    _demo()
