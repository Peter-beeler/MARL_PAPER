"""
Commons Harvest environment — Tragedy of the Commons.

Legend:
- '*' = land (all cells)
- 'a' = apple (resource)
- '1'..'9' = agents (rendered over cells)

Scoring:
- eat_reward points for eating one apple (default 1.0)

Apple regrowth depends on Moore neighborhood density (8 surrounding cells):
- 0 neighboring apples: 0% spawn chance (dead zone)
- 1-2 neighbors: low chance
- 3-5+ neighbors: high chance

If agents over-harvest and deplete an area, regrowth drops to zero permanently
for that patch (tragedy of the commons).

This file implements HarvestEnv with a minimal API:
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
    max_steps: int = 500
    seed: Optional[int] = 42

    # Apple initial placement
    init_apple_density: float = 0.35  # fraction of cells starting with apples
    max_apples: Optional[int] = None  # cap on total apples (None = unlimited)

    # Regrowth probabilities indexed by Moore neighborhood apple count.
    # regrowth_prob_N = probability an empty cell spawns an apple when it has
    # exactly N apple neighbors (capped at 5 for 5+).
    regrowth_prob_0: float = 0.0     # 0 apple neighbors — dead zone
    regrowth_prob_1: float = 0.005   # 1 neighbor
    regrowth_prob_2: float = 0.01    # 2 neighbors
    regrowth_prob_3: float = 0.025   # 3 neighbors
    regrowth_prob_4: float = 0.05    # 4 neighbors
    regrowth_prob_5: float = 0.1     # 5+ neighbors

    # Rewards
    eat_reward: float = 1.0

Action = int  # 0: stay, 1: eat, 2: up, 3: down, 4: left, 5: right


class HarvestEnv:
    """Commons Harvest grid environment.

    Grid layers:
      - terrain[y][x]: '*' (all land)
      - items[y][x]: None | 'a' (apple)
      - agents: dict of agent_id -> (x, y)
    Rendering overlays agents over items over terrain.
    """

    # Action vocabulary (both string and int supported)
    ACTIONS = {
        0: "stay", 1: "eat", 2: "up", 3: "down", 4: "left", 5: "right",
        "stay": "stay", "eat": "eat",
        "up": "up", "down": "down", "left": "left", "right": "right"
    }

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.rng = random.Random(self.cfg.seed)

        self.width = self.cfg.width
        self.height = self.cfg.height

        # Layers
        self.terrain: List[List[str]] = []  # all '*'
        self.items: List[List[Optional[str]]] = []  # None or 'a'
        self.agents: Dict[int, Tuple[int, int]] = {}
        self.prev_agents: Dict[int, Tuple[int, int]] = {}

        # Episode state
        self.step_count = 0
        self.scores: Dict[int, float] = {}

        self.reset()

    # ---------------------------
    # Core API
    # ---------------------------
    def reset(self):
        """Reset the environment and return the initial observation."""
        self.step_count = 0
        self.scores = {i: 0.0 for i in self._agent_ids}

        # All land terrain
        self.terrain = [["*" for _ in range(self.width)] for _ in range(self.height)]

        # Empty items layer, then scatter apples
        self.items = [[None for _ in range(self.width)] for _ in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if self.rng.random() < self.cfg.init_apple_density:
                    self.items[y][x] = 'a'

        # Place agents on random empty cells
        self.agents = {}
        empty_cells = [(x, y) for y in range(self.height)
                       for x in range(self.width) if self.items[y][x] is None]
        self.rng.shuffle(empty_cells)
        if len(empty_cells) < self.cfg.n_agents:
            all_cells = [(x, y) for y in range(self.height) for x in range(self.width)]
            self.rng.shuffle(all_cells)
            empty_cells = all_cells

        for i in range(1, self.cfg.n_agents + 1):
            self.agents[i] = empty_cells[i - 1]

        self.prev_agents = {i: pos for i, pos in self.agents.items()}
        return self._observation()

    def step(self, actions: Dict[int, Action | str]):
        """Advance the environment by one step.

        actions: mapping of agent_id -> action (int or str). Missing IDs default to stay.

        Returns: (observation, rewards, done, info)
        """
        self.step_count += 1
        rewards = {i: 0.0 for i in self._agent_ids}

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

            if a_name in ("stay", "eat"):
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

        # Eat interactions
        for i, (x, y) in self.agents.items():
            if self.items[y][x] == 'a' and chosen_action.get(i) == "eat":
                rewards[i] += self.cfg.eat_reward
                self.scores[i] += self.cfg.eat_reward
                self.items[y][x] = None

        # Regrowth phase
        self._regrowth()

        done = self.step_count >= self.cfg.max_steps
        obs = self._observation()
        info = {
            "step": self.step_count,
            "scores": dict(self.scores),
            "apple_count": self._count_items('a'),
        }
        return obs, rewards, done, info

    def render(self) -> str:
        """Return a human-readable string of the current grid."""
        grid_chars = [[self.terrain[y][x] for x in range(self.width)]
                      for y in range(self.height)]
        for y in range(self.height):
            for x in range(self.width):
                if self.items[y][x] is not None:
                    grid_chars[y][x] = self.items[y][x]
        for i, (x, y) in self.agents.items():
            grid_chars[y][x] = str(i % 10)
        return "\n".join("".join(row) for row in grid_chars)

    def get_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current environment state."""
        return {
            "terrain": [row[:] for row in self.terrain],
            "items": [row[:] for row in self.items],
            "agents": {aid: (pos[0], pos[1]) for aid, pos in self.agents.items()},
            "prev_agents": {aid: (pos[0], pos[1]) for aid, pos in self.prev_agents.items()},
            "scores": dict(self.scores),
            "step_count": self.step_count,
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by get_state()."""
        self.terrain = [row[:] for row in state["terrain"]]
        self.items = [row[:] for row in state["items"]]
        self.agents = {int(aid): (pos[0], pos[1]) for aid, pos in state["agents"].items()}
        if "prev_agents" in state:
            self.prev_agents = {int(aid): (pos[0], pos[1])
                                for aid, pos in state["prev_agents"].items()}
        else:
            self.prev_agents = {int(aid): (pos[0], pos[1])
                                for aid, pos in state["agents"].items()}
        self.scores = {int(aid): float(score) for aid, score in state["scores"].items()}
        self.step_count = int(state["step_count"])
        self.rng.setstate(state["rng_state"])

    # ---------------------------
    # Helpers
    # ---------------------------
    @property
    def _agent_ids(self) -> List[int]:
        return list(range(1, self.cfg.n_agents + 1))

    def _count_items(self, token: str) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width)
                   if self.items[y][x] == token)

    def _moore_apple_count(self, x: int, y: int) -> int:
        """Count apples in the Moore neighborhood (8 surrounding cells)."""
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.items[ny][nx] == 'a':
                        count += 1
        return count

    def _regrowth_prob(self, neighbor_count: int) -> float:
        """Get regrowth probability for a given number of apple neighbors."""
        probs = [
            self.cfg.regrowth_prob_0,
            self.cfg.regrowth_prob_1,
            self.cfg.regrowth_prob_2,
            self.cfg.regrowth_prob_3,
            self.cfg.regrowth_prob_4,
            self.cfg.regrowth_prob_5,
        ]
        return probs[min(neighbor_count, len(probs) - 1)]

    def _regrowth(self):
        """Spawn new apples based on Moore neighborhood density."""
        if (self.cfg.max_apples is not None
                and self._count_items('a') >= self.cfg.max_apples):
            return

        for y in range(self.height):
            for x in range(self.width):
                if self.items[y][x] is not None:
                    continue
                if any((x, y) == pos for pos in self.agents.values()):
                    continue
                n_count = self._moore_apple_count(x, y)
                prob = self._regrowth_prob(n_count)
                if prob > 0 and self.rng.random() < prob:
                    self.items[y][x] = 'a'
                    if (self.cfg.max_apples is not None
                            and self._count_items('a') >= self.cfg.max_apples):
                        return

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
        for yy, y in enumerate(range(y0, y1 + 1)):
            for xx, x in enumerate(range(x0, x1 + 1)):
                if self.items[y][x] is not None:
                    grid_chars[yy][xx] = self.items[y][x]
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

    def _find_nearest_items(self, agent_id: int, item_type: str, n: int = 3,
                            exclude_local_window: bool = False) -> List[Tuple[int, int]]:
        """Find N nearest items of a specific type for an agent."""
        agent_pos = self.agents[agent_id]
        ax, ay = agent_pos

        item_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.items[y][x] == item_type:
                    if exclude_local_window:
                        half_w, half_h = 2, 2
                        if abs(x - ax) <= half_w and abs(y - ay) <= half_h:
                            continue
                    item_positions.append((x, y))

        items_with_distance = [
            (pos, self._calculate_distance(agent_pos, pos))
            for pos in item_positions
        ]
        items_with_distance.sort(key=lambda x: x[1])
        return [pos for pos, dist in items_with_distance[:n]]


# ---------------------------
# Quick demo
# ---------------------------
def _demo():
    cfg = Config(width=16, height=16, n_agents=4, max_steps=30,
                 init_apple_density=0.3)
    env = HarvestEnv(cfg)
    print("=== Commons Harvest Demo ===")
    print("Initial grid:")
    print(env.render())
    print(f"Apples: {env._count_items('a')}")
    print()

    while True:
        actions = {i: env.rng.choice([0, 1, 2, 3, 4, 5]) for i in env._agent_ids}
        obs, rewards, done, info = env.step(actions)
        if env.step_count % 5 == 0 or done:
            print(f"Step {info['step']}: apples={info['apple_count']} "
                  f"scores={info['scores']}")
            print(env.render())
            print()
        if done:
            break

    print("Final scores:", info['scores'])
    print("Total score:", sum(info['scores'].values()))


if __name__ == "__main__":
    _demo()
