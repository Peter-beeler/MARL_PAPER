"""
Simple 2D grid cleanup game environment with directional movement.

Legend:
- '*' = land
- 'x' = water (river)
- '#' = dirt (only on water)
- 'a' = apple (only on land)
- '1'..'9' = agents (rendered over cells)

Scoring:
- clean_reward points for cleaning one dirt ('#') (default 0.0, configurable via Config)
- eat_reward points for eating one apple ('a') (default 1.0, configurable via Config)

Apples auto-generate on land. By default, apples only begin spawning once the
current dirt count drops below the initial dirt count at reset; as it gets
cleaner, apples spawn faster.

This file implements a single class `CleanupEnvMove` with a minimal API:
- reset() -> observation
- step(actions: dict[int, str|int]) -> (obs, rewards, done, info)
- render() -> str

Difference from env.py: Agents use directional movement (up/down/left/right)
instead of teleportation.

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
    width: int = 15
    height: int = 9
    n_agents: int = 3  # supports up to 9 for rendering with digits

    # River geometry
    river_width: int = 3  # centered vertical band of water columns

    # Episode control
    max_steps: int = 200
    seed: Optional[int] = 42

    # Spawning probabilities (per empty cell, per step)
    dirt_spawn_prob: float = 0.02  # on water cells
    # Apple spawn base rate, scaled by cleanliness in step()
    apple_spawn_base: float = 0.05  # on land cells
    apple_cleanliness_exponent: float = 1.0  # nonlinearity: p = base * clean**exp

    # Limits
    max_apples: Optional[int] = 5  # cap apples present on grid
    max_dirts: Optional[int] = None   # None => unlimited

    # Rewards
    eat_reward: float = 1.0    # reward for eating an apple
    clean_reward: float = 0.0  # reward for cleaning a dirt tile (0 = no immediate reward)

    # Initial dirt placement (on reset)
    init_dirt_prob: float = 0.35  # probability per water cell to start as dirt

Action = int  # 0: stay, 1: eat, 2: clean, 3: up, 4: down, 5: left, 6: right


class CleanupEnvMove:
    """A simple grid environment with agents cleaning a river and eating apples.

    Grid layers:
      - terrain[y][x]: '*' (land) or 'x' (water)
      - items[y][x]: None | 'a' (apple on land) | '#' (dirt on water)
      - agents: dict of agent_id -> (x, y)
    Rendering overlays agents over items over terrain.
    """

    # Action vocabulary (both string and int supported)
    ACTIONS = {
        0: "stay", 1: "eat", 2: "clean", 3: "up", 4: "down", 5: "left", 6: "right",
        "stay": "stay", "eat": "eat", "clean": "clean",
        "up": "up", "down": "down", "left": "left", "right": "right"
    }

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg or Config()
        self.rng = random.Random(self.cfg.seed)

        self.width = self.cfg.width
        self.height = self.cfg.height

        # Layers
        self.terrain: List[List[str]] = []  # '*' or 'x'
        self.items: List[List[Optional[str]]] = []  # None, 'a', '#'
        self.agents: Dict[int, Tuple[int, int]] = {}
        self.prev_agents: Dict[int, Tuple[int, int]] = {}  # Previous positions for movement tracking

        # Episode state
        self.step_count = 0
        self.scores: Dict[int, float] = {}
        self.init_dirt_count: int = 0

        self.reset()

    # ---------------------------
    # Core API
    # ---------------------------
    def reset(self):
        """Reset the environment and return the initial observation (grid string)."""
        self.step_count = 0
        self.scores = {i: 0.0 for i in self._agent_ids}

        # Create terrain with a centered vertical river
        self.terrain = [["*" for _ in range(self.width)] for _ in range(self.height)]
        river_left = max(0, (self.width - self.cfg.river_width) // 2)
        river_right = min(self.width - 1, river_left + self.cfg.river_width - 1)
        for y in range(self.height):
            for x in range(self.width):
                if river_left <= x <= river_right:
                    self.terrain[y][x] = "x"  # water

        # Empty items layer
        self.items = [[None for _ in range(self.width)] for _ in range(self.height)]

        # Place initial dirt on water according to init_dirt_prob
        initial_dirts = 0
        for y in range(self.height):
            # Early stop if reached cap
            if self.cfg.max_dirts is not None and initial_dirts >= self.cfg.max_dirts:
                break
            for x in range(self.width):
                if self.terrain[y][x] == 'x' and self.rng.random() < self.cfg.init_dirt_prob:
                    if self.cfg.max_dirts is not None and initial_dirts >= self.cfg.max_dirts:
                        break
                    self.items[y][x] = '#'
                    initial_dirts += 1
        self.init_dirt_count = initial_dirts

        # Place agents on random land cells
        # Try to place agents where they can see at least one dirt or apple
        self.agents = {}
        land_cells = [(x, y) for y in range(self.height) for x in range(self.width) if self.terrain[y][x] == "*"]
        self.rng.shuffle(land_cells)
        if len(land_cells) < self.cfg.n_agents:
            raise ValueError("Not enough land cells to place all agents.")

        # First, try to find positions with visible items
        good_positions = []
        for pos in land_cells:
            if self._has_visible_items(pos):
                good_positions.append(pos)

        # Place agents: prioritize good positions, fall back to any land cell
        for i in range(1, self.cfg.n_agents + 1):
            if good_positions:
                self.agents[i] = good_positions.pop(0)
            else:
                # Fall back to remaining land cells if no good positions left
                remaining = [pos for pos in land_cells if pos not in self.agents.values()]
                if remaining:
                    self.agents[i] = remaining[0]
                else:
                    # Last resort: reuse land_cells
                    self.agents[i] = land_cells[i - 1]

        # Initialize previous positions (same as current for first step)
        self.prev_agents = {i: pos for i, pos in self.agents.items()}

        return self._observation()

    def step(self, actions: Dict[int, Action | str]):
        """Advance the environment by one step.

        actions: mapping of agent_id -> action (int or str). Missing IDs default to stay.

        Returns: (observation, rewards, done, info)
        - observation: current grid string after updates
        - rewards: dict of agent_id -> float
        - done: bool, episode termination by max_steps
        - info: dict with diagnostics
        """
        self.step_count += 1
        rewards = {i: 0.0 for i in self._agent_ids}

        # Compute desired moves with directional movement
        desired: Dict[int, Tuple[int, int]] = {}
        chosen_action: Dict[int, str] = {}
        for i in self._agent_ids:
            ax, ay = self.agents[i]
            a_raw = actions.get(i, 0)
            a_name = self.ACTIONS.get(a_raw if a_raw in self.ACTIONS else (a_raw.lower() if isinstance(a_raw, str) else a_raw), "stay")
            chosen_action[i] = a_name

            if a_name == "stay":
                desired[i] = (ax, ay)
            elif a_name == "eat":
                # Stay in place; eating happens after movement
                desired[i] = (ax, ay)
            elif a_name == "clean":
                # Stay in place; cleaning happens after movement
                desired[i] = (ax, ay)
            elif a_name == "up":
                # Move up (decrease internal y, toward top of grid)
                # Display coordinates show this as increasing y
                new_y = max(0, ay - 1)
                desired[i] = (ax, new_y)
            elif a_name == "down":
                # Move down (increase internal y, toward bottom of grid)
                # Display coordinates show this as decreasing y
                new_y = min(self.height - 1, ay + 1)
                desired[i] = (ax, new_y)
            elif a_name == "left":
                # Move left (decrease x)
                new_x = max(0, ax - 1)
                desired[i] = (new_x, ay)
            elif a_name == "right":
                # Move right (increase x)
                new_x = min(self.width - 1, ax + 1)
                desired[i] = (new_x, ay)
            else:
                # Unknown action, default to stay
                desired[i] = (ax, ay)

        # Resolve collisions: agents moving to same cell all stay
        target_counts: Dict[Tuple[int, int], int] = {}
        for pos in desired.values():
            target_counts[pos] = target_counts.get(pos, 0) + 1

        new_positions: Dict[int, Tuple[int, int]] = {}
        occupied_targets: set[Tuple[int, int]] = set(self.agents.values())
        for i in self._agent_ids:
            pos = desired[i]
            # If multiple agents want same target, they all stay
            if target_counts[pos] > 1 and pos != self.agents[i]:
                new_positions[i] = self.agents[i]
                continue
            # If target already occupied by another agent that is staying there, block
            if pos in new_positions.values():
                new_positions[i] = self.agents[i]
                continue
            new_positions[i] = pos

        # Save previous positions before updating (for movement tracking)
        self.prev_agents = {i: pos for i, pos in self.agents.items()}
        self.agents = new_positions

        # Interactions: action-gated eat/clean
        for i, (x, y) in self.agents.items():
            item = self.items[y][x]
            if item == 'a' and chosen_action.get(i) == "eat":
                rewards[i] += self.cfg.eat_reward
                self.scores[i] += self.cfg.eat_reward
                self.items[y][x] = None
            elif item == '#' and chosen_action.get(i) == "clean":
                rewards[i] += self.cfg.clean_reward
                self.scores[i] += self.cfg.clean_reward
                self.items[y][x] = None

        # Spawn phase
        self._spawn_dirt()
        self._spawn_apples()

        done = self.step_count >= self.cfg.max_steps
        obs = self._observation()
        info = {
            "step": self.step_count,
            "scores": dict(self.scores),
            "dirt_count": self._count_items('#'),
            "apple_count": self._count_items('a'),
            "init_dirt_count": self.init_dirt_count,
        }
        return obs, rewards, done, info

    def render(self) -> str:
        """Return a human-readable string of the current grid."""
        grid_chars = [[self.terrain[y][x] for x in range(self.width)] for y in range(self.height)]
        # overlay items
        for y in range(self.height):
            for x in range(self.width):
                if self.items[y][x] is not None:
                    grid_chars[y][x] = self.items[y][x]
        # overlay agents (digits)
        for i, (x, y) in self.agents.items():
            ch = str(i % 10)  # supports up to 9 unique; wraps beyond to keep char
            grid_chars[y][x] = ch
        lines = ["".join(row) for row in grid_chars]
        return "\n".join(lines)

    def get_state(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the current environment state."""
        return {
            "terrain": [row[:] for row in self.terrain],
            "items": [row[:] for row in self.items],
            "agents": {aid: (pos[0], pos[1]) for aid, pos in self.agents.items()},
            "prev_agents": {aid: (pos[0], pos[1]) for aid, pos in self.prev_agents.items()},
            "scores": dict(self.scores),
            "step_count": self.step_count,
            "init_dirt_count": self.init_dirt_count,
            "rng_state": self.rng.getstate(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore a snapshot produced by get_state()."""
        self.terrain = [row[:] for row in state["terrain"]]
        self.items = [row[:] for row in state["items"]]
        self.agents = {int(aid): (pos[0], pos[1]) for aid, pos in state["agents"].items()}
        # Handle prev_agents (may not exist in older saved states)
        if "prev_agents" in state:
            self.prev_agents = {int(aid): (pos[0], pos[1]) for aid, pos in state["prev_agents"].items()}
        else:
            self.prev_agents = {int(aid): (pos[0], pos[1]) for aid, pos in state["agents"].items()}
        self.scores = {int(aid): float(score) for aid, score in state["scores"].items()}
        self.step_count = int(state["step_count"])
        self.init_dirt_count = int(state["init_dirt_count"])
        self.rng.setstate(state["rng_state"])

    # ---------------------------
    # Helpers
    # ---------------------------
    @property
    def _agent_ids(self) -> List[int]:
        return list(range(1, self.cfg.n_agents + 1))

    def _count_items(self, token: str) -> int:
        return sum(1 for y in range(self.height) for x in range(self.width) if self.items[y][x] == token)

    def get_agent_movement(self, agent_id: int) -> Tuple[str, Tuple[int, int]]:
        """Get the movement direction and previous position for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            Tuple of (direction_string, prev_position_display_coords)
            direction_string is one of: "UP", "DOWN", "LEFT", "RIGHT", "STAYED"
            prev_position uses display coordinates (y=0 at bottom)
        """
        if agent_id not in self.agents or agent_id not in self.prev_agents:
            return ("STAYED", (0, 0))

        curr_x, curr_y = self.agents[agent_id]
        prev_x, prev_y = self.prev_agents[agent_id]

        # Compute movement direction (in internal coordinates)
        dx = curr_x - prev_x
        dy = curr_y - prev_y

        if dx == 0 and dy == 0:
            direction = "STAYED"
        elif dy < 0:  # Moving up in internal coords (toward row 0)
            direction = "UP"
        elif dy > 0:  # Moving down in internal coords
            direction = "DOWN"
        elif dx < 0:
            direction = "LEFT"
        elif dx > 0:
            direction = "RIGHT"
        else:
            direction = "STAYED"

        # Convert previous position to display coordinates (y=0 at bottom)
        prev_y_display = (self.height - 1) - prev_y
        return (direction, (prev_x, prev_y_display))

    def _local_render(self, agent_id: int) -> str:
        """Render a 5x3 local observation for the given agent."""
        ax, ay = self.agents[agent_id]
        half_w = 2
        half_h = 1
        y0 = max(0, ay - half_h)
        y1 = min(self.height - 1, ay + half_h)
        x0 = max(0, ax - half_w)
        x1 = min(self.width - 1, ax + half_w)

        grid_chars = [[self.terrain[y][x] for x in range(x0, x1 + 1)] for y in range(y0, y1 + 1)]
        # overlay items
        for yy, y in enumerate(range(y0, y1 + 1)):
            for xx, x in enumerate(range(x0, x1 + 1)):
                if self.items[y][x] is not None:
                    grid_chars[yy][xx] = self.items[y][x]
        # overlay agents in the local window
        for aid, (x, y) in self.agents.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                grid_chars[y - y0][x - x0] = str(aid % 10)
        return "\n".join("".join(row) for row in grid_chars)

    def _spawn_dirt(self):
        # Optional cap
        if self.cfg.max_dirts is not None and self._count_items('#') >= self.cfg.max_dirts:
            return
        for y in range(self.height):
            for x in range(self.width):
                if self.terrain[y][x] != 'x':
                    continue  # only water
                if self.items[y][x] is not None:
                    continue
                # avoid spawning under an agent to make cleaning an action
                if any((x, y) == pos for pos in self.agents.values()):
                    continue
                if self.rng.random() < self.cfg.dirt_spawn_prob:
                    self.items[y][x] = '#'

    def _spawn_apples(self):
        # Apple spawn rate depends on improvement relative to initial dirt level.
        # If current dirt >= initial dirt, no apples. As dirt decreases below initial, apples appear with increasing rate.
        total_water = sum(1 for y in range(self.height) for x in range(self.width) if self.terrain[y][x] == 'x')
        dirt_count = self._count_items('#')
        if self.init_dirt_count > 0:
            if dirt_count >= self.init_dirt_count:
                rate = 0.0
            else:
                progress = 1.0 - (dirt_count / self.init_dirt_count)
                progress = max(0.0, min(1.0, progress))
                rate = self.cfg.apple_spawn_base * (progress ** self.cfg.apple_cleanliness_exponent)
        else:
            # If no initial dirt, fall back to cleanliness-based spawning
            cleanliness = 1.0
            if total_water > 0:
                cleanliness = max(0.0, 1.0 - dirt_count / total_water)
            rate = self.cfg.apple_spawn_base * (cleanliness ** self.cfg.apple_cleanliness_exponent)

        # Optional cap
        if self.cfg.max_apples is not None and self._count_items('a') >= self.cfg.max_apples:
            return

        # Collect all dirt positions to avoid spawning apples in their observation windows
        dirt_positions = []
        for dy in range(self.height):
            for dx in range(self.width):
                if self.items[dy][dx] == '#':
                    dirt_positions.append((dx, dy))

        half_w = 2  # local window width=5
        half_h = 1  # local window height=3

        for y in range(self.height):
            for x in range(self.width):
                if self.terrain[y][x] != '*':
                    continue  # only land
                if self.items[y][x] is not None:
                    continue
                # avoid spawning where an agent stands
                if any((x, y) == pos for pos in self.agents.values()):
                    continue
                # avoid spawning within observation window of any dirt
                if any(abs(x - dx) <= half_w and abs(y - dy) <= half_h
                       for dx, dy in dirt_positions):
                    continue
                if self.rng.random() < rate:
                    self.items[y][x] = 'a'

    def _has_visible_items(self, pos: Tuple[int, int]) -> bool:
        """Check if there are any items (dirt or apples) visible from this position.

        This is used during initialization to place agents in better starting positions.
        """
        x, y = pos
        half_w = 2  # local window width=5
        half_h = 1  # local window height=3
        y0 = max(0, y - half_h)
        y1 = min(self.height - 1, y + half_h)
        x0 = max(0, x - half_w)
        x1 = min(self.width - 1, x + half_w)

        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                if self.items[yy][xx] in ('a', '#'):
                    return True
        return False

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions."""
        x1, y1 = pos1
        x2, y2 = pos2
        return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

    def _find_nearest_items(self, agent_id: int, item_type: str, n: int = 3,
                            exclude_local_window: bool = False) -> List[Tuple[int, int]]:
        """Find N nearest items of a specific type for an agent.

        Args:
            agent_id: Agent ID
            item_type: '#' for dirt or 'a' for apple
            n: Number of nearest items to return
            exclude_local_window: If True, exclude items in agent's 5×3 local window

        Returns:
            List of (x, y) tuples for nearest items, sorted by distance
        """
        agent_pos = self.agents[agent_id]
        ax, ay = agent_pos

        # Collect all items of the specified type
        item_positions = []
        for y in range(self.height):
            for x in range(self.width):
                if self.items[y][x] == item_type:
                    # Skip if in local window and exclusion is enabled
                    if exclude_local_window:
                        half_w, half_h = 2, 1
                        if abs(x - ax) <= half_w and abs(y - ay) <= half_h:
                            continue
                    item_positions.append((x, y))

        # Calculate distances and sort
        items_with_distance = [
            (pos, self._calculate_distance(agent_pos, pos))
            for pos in item_positions
        ]
        items_with_distance.sort(key=lambda x: x[1])

        # Return top N positions
        return [pos for pos, dist in items_with_distance[:n]]

    def _observation(self):
        # Return per-agent local observations (5x3 window)
        return {i: self._local_render(i) for i in self._agent_ids}


# ---------------------------
# Quick demo
# ---------------------------
def _demo():
    cfg = Config(width=15, height=9, n_agents=3, river_width=3, max_steps=30,
                 dirt_spawn_prob=0.05, apple_spawn_base=0.08, apple_cleanliness_exponent=1.0)
    env = CleanupEnvMove(cfg)
    print("Initial (global):")
    print(env.render())
    print()

    while True:
        # random actions among stay/eat/clean/up/down/left/right
        actions = {i: env.rng.choice([0, 1, 2, 3, 4, 5, 6]) for i in env._agent_ids}
        obs, rewards, done, info = env.step(actions)
        print(env.render())
        print("local obs examples:")
        for i in env._agent_ids:
            print(f"Agent {i} local:\n{obs[i]}")
        print(f"rewards={rewards} scores={info['scores']} step={info['step']} dirt={info['dirt_count']} apples={info['apple_count']}")
        if done:
            break


if __name__ == "__main__":
    _demo()
