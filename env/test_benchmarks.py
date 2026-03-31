"""
Benchmark tests for MARL environments.

Tests:
1. Greedy vs Cooperative policy on Commons Harvest
   - Greedy agents eat every apple → resource depletion → low total score
   - Cooperative agents leave seed apples → sustained regrowth → higher score

2. Random vs Coordinated policy on Stag Hunt
   - Random: agents rarely coordinate → few stag catches
   - Coordinated: paired agents target stags together → more stag catches
"""

from __future__ import annotations

import sys
import os

# Add parent directory to path so we can import env modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.harvest_env import HarvestEnv, Config as HarvestConfig
from env.staghunt_env import StagHuntEnv, Config as StagHuntConfig


# ============================================================
# Harvest Policies
# ============================================================

def greedy_harvest_action(env: HarvestEnv, agent_id: int) -> int:
    """Greedy: always eat if on apple, otherwise move toward nearest apple."""
    ax, ay = env.agents[agent_id]
    if env.items[ay][ax] == 'a':
        return 1  # eat

    nearest = env._find_nearest_items(agent_id, 'a', n=1)
    if nearest:
        tx, ty = nearest[0]
        dx, dy = tx - ax, ty - ay
        if abs(dx) >= abs(dy):
            return 5 if dx > 0 else 4  # right / left
        else:
            return 3 if dy > 0 else 2  # down / up
    return 0  # stay


def cooperative_harvest_action(env: HarvestEnv, agent_id: int) -> int:
    """Cooperative: only eat when Moore neighborhood has >= 2 other apples (leave seeds)."""
    ax, ay = env.agents[agent_id]
    if env.items[ay][ax] == 'a':
        neighbor_count = env._moore_apple_count(ax, ay)
        if neighbor_count >= 2:
            return 1  # eat — enough seeds remain for regrowth
        else:
            # Don't eat, move away to find a richer patch
            nearest = env._find_nearest_items(agent_id, 'a', n=3)
            # Find an apple that has rich neighbors
            for tx, ty in nearest:
                if (tx, ty) == (ax, ay):
                    continue
                if env._moore_apple_count(tx, ty) >= 2:
                    dx, dy = tx - ax, ty - ay
                    if abs(dx) >= abs(dy):
                        return 5 if dx > 0 else 4
                    else:
                        return 3 if dy > 0 else 2
            return 0  # stay if no good target

    # Not on an apple — move toward nearest
    nearest = env._find_nearest_items(agent_id, 'a', n=1)
    if nearest:
        tx, ty = nearest[0]
        dx, dy = tx - ax, ty - ay
        if abs(dx) >= abs(dy):
            return 5 if dx > 0 else 4
        else:
            return 3 if dy > 0 else 2
    return 0


# ============================================================
# Stag Hunt Policies
# ============================================================

def random_staghunt_action(env: StagHuntEnv, agent_id: int) -> int:
    """Random: pick a random action each step."""
    return env.rng.choice([0, 1, 2, 3, 4, 5])


def coordinated_staghunt_action(env: StagHuntEnv, agent_id: int,
                                 partner_id: int) -> int:
    """Coordinated: pair with a partner, both move toward nearest stag and hunt."""
    ax, ay = env.agents[agent_id]

    # Find nearest alive stag
    nearest_stags = env.find_nearest_prey(agent_id, "stag", n=1)
    if nearest_stags:
        _, (sx, sy) = nearest_stags[0]
        dist = abs(ax - sx) + abs(ay - sy)
        if dist <= 1:
            return 1  # hunt — we're adjacent
        # Move toward stag
        dx, dy = sx - ax, sy - ay
        if abs(dx) >= abs(dy):
            return 5 if dx > 0 else 4  # right / left
        else:
            return 3 if dy > 0 else 2  # down / up

    # No stags alive — hunt nearest hare as fallback
    nearest_hares = env.find_nearest_prey(agent_id, "hare", n=1)
    if nearest_hares:
        _, (hx, hy) = nearest_hares[0]
        dist = abs(ax - hx) + abs(ay - hy)
        if dist <= 1:
            return 1  # hunt
        dx, dy = hx - ax, hy - ay
        if abs(dx) >= abs(dy):
            return 5 if dx > 0 else 4
        else:
            return 3 if dy > 0 else 2

    return 0  # stay


# ============================================================
# Test 1: Harvest — Greedy vs Cooperative
# ============================================================

def test_harvest_greedy_vs_cooperative():
    """Verify that greedy harvesting depletes resources while cooperative sustains them."""
    print("=" * 60)
    print("TEST 1: Commons Harvest — Greedy vs Cooperative")
    print("=" * 60)

    cfg = HarvestConfig(
        width=16, height=16, n_agents=4, max_steps=300,
        seed=42, init_apple_density=0.35,
    )

    # --- Greedy run ---
    env_greedy = HarvestEnv(cfg)
    greedy_apple_history = []
    for step in range(cfg.max_steps):
        actions = {i: greedy_harvest_action(env_greedy, i) for i in env_greedy._agent_ids}
        obs, rewards, done, info = env_greedy.step(actions)
        greedy_apple_history.append(info['apple_count'])
        if done:
            break
    greedy_total = sum(info['scores'].values())
    greedy_final_apples = info['apple_count']

    print(f"\nGreedy Policy:")
    print(f"  Total score: {greedy_total:.0f}")
    print(f"  Final apples on grid: {greedy_final_apples}")
    print(f"  Apple count at step 50: {greedy_apple_history[min(49, len(greedy_apple_history)-1)]}")
    print(f"  Apple count at step 150: {greedy_apple_history[min(149, len(greedy_apple_history)-1)]}")

    # --- Cooperative run ---
    env_coop = HarvestEnv(cfg)
    coop_apple_history = []
    for step in range(cfg.max_steps):
        actions = {i: cooperative_harvest_action(env_coop, i) for i in env_coop._agent_ids}
        obs, rewards, done, info = env_coop.step(actions)
        coop_apple_history.append(info['apple_count'])
        if done:
            break
    coop_total = sum(info['scores'].values())
    coop_final_apples = info['apple_count']

    print(f"\nCooperative Policy:")
    print(f"  Total score: {coop_total:.0f}")
    print(f"  Final apples on grid: {coop_final_apples}")
    print(f"  Apple count at step 50: {coop_apple_history[min(49, len(coop_apple_history)-1)]}")
    print(f"  Apple count at step 150: {coop_apple_history[min(149, len(coop_apple_history)-1)]}")

    # Verification
    print(f"\n--- Verification ---")
    # The cooperative policy should maintain more apples on the grid
    # (sustained resource) leading to higher or comparable total collection
    # over a long episode.
    greedy_late_apples = sum(greedy_apple_history[150:]) / max(1, len(greedy_apple_history[150:]))
    coop_late_apples = sum(coop_apple_history[150:]) / max(1, len(coop_apple_history[150:]))
    print(f"  Avg apples on grid (steps 150-300): greedy={greedy_late_apples:.1f}, coop={coop_late_apples:.1f}")

    assert coop_late_apples > greedy_late_apples, (
        f"FAIL: Cooperative should sustain more apples late-game "
        f"({coop_late_apples:.1f} vs {greedy_late_apples:.1f})")
    print("  PASS: Cooperative policy sustains more apples than greedy in late game.")

    return greedy_total, coop_total


# ============================================================
# Test 2: Stag Hunt — Random vs Coordinated
# ============================================================

def test_staghunt_random_vs_coordinated():
    """Verify that coordination is required to catch stags."""
    print("\n" + "=" * 60)
    print("TEST 2: Stag Hunt — Random vs Coordinated")
    print("=" * 60)

    cfg = StagHuntConfig(
        width=12, height=12, n_agents=4, n_stags=2, n_hares=4,
        max_steps=200, seed=42,
        stag_reward=5.0, hare_reward=1.0,
        stag_move_prob=0.2, hare_move_prob=0.4,
        respawn_delay=8,
    )

    # --- Random policy run ---
    env_rand = StagHuntEnv(cfg)
    for step in range(cfg.max_steps):
        actions = {i: random_staghunt_action(env_rand, i) for i in env_rand._agent_ids}
        obs, rewards, done, info = env_rand.step(actions)
        if done:
            break
    rand_stag_catches = info['total_stag_catches']
    rand_hare_catches = info['total_hare_catches']
    rand_total = sum(info['scores'].values())

    print(f"\nRandom Policy:")
    print(f"  Total score: {rand_total:.0f}")
    print(f"  Stag catches: {rand_stag_catches}")
    print(f"  Hare catches: {rand_hare_catches}")

    # --- Coordinated policy run ---
    # Pair agents: (1,2) and (3,4) target stags together
    env_coord = StagHuntEnv(cfg)
    pairs = {1: 2, 2: 1, 3: 4, 4: 3}
    for step in range(cfg.max_steps):
        actions = {}
        for i in env_coord._agent_ids:
            partner = pairs[i]
            actions[i] = coordinated_staghunt_action(env_coord, i, partner)
        obs, rewards, done, info = env_coord.step(actions)
        if done:
            break
    coord_stag_catches = info['total_stag_catches']
    coord_hare_catches = info['total_hare_catches']
    coord_total = sum(info['scores'].values())

    print(f"\nCoordinated Policy:")
    print(f"  Total score: {coord_total:.0f}")
    print(f"  Stag catches: {coord_stag_catches}")
    print(f"  Hare catches: {coord_hare_catches}")

    # Verification
    print(f"\n--- Verification ---")
    assert coord_stag_catches > rand_stag_catches, (
        f"FAIL: Coordinated should catch more stags "
        f"({coord_stag_catches} vs {rand_stag_catches})")
    print(f"  PASS: Coordinated catches more stags ({coord_stag_catches} vs {rand_stag_catches}).")

    assert coord_total > rand_total, (
        f"FAIL: Coordinated should score higher "
        f"({coord_total:.0f} vs {rand_total:.0f})")
    print(f"  PASS: Coordinated scores higher ({coord_total:.0f} vs {rand_total:.0f}).")

    return rand_total, coord_total


# ============================================================
# Test 3: Basic API smoke tests
# ============================================================

def test_api_smoke():
    """Verify reset/step/render/get_state/set_state work for all envs."""
    print("\n" + "=" * 60)
    print("TEST 3: API Smoke Tests")
    print("=" * 60)

    # Harvest
    henv = HarvestEnv(HarvestConfig(width=8, height=8, n_agents=2, max_steps=10, seed=1))
    obs = henv.reset()
    assert isinstance(obs, dict) and len(obs) == 2
    state = henv.get_state()
    obs2, rew, done, info = henv.step({1: 0, 2: 1})
    assert isinstance(rew, dict) and isinstance(info, dict)
    rendered = henv.render()
    assert isinstance(rendered, str) and len(rendered) > 0
    henv.set_state(state)
    assert henv.step_count == state["step_count"]
    print("  Harvest API: PASS")

    # Stag Hunt
    senv = StagHuntEnv(StagHuntConfig(width=8, height=8, n_agents=2,
                                       n_stags=1, n_hares=2, max_steps=10, seed=1))
    obs = senv.reset()
    assert isinstance(obs, dict) and len(obs) == 2
    state = senv.get_state()
    obs2, rew, done, info = senv.step({1: 0, 2: 1})
    assert isinstance(rew, dict) and isinstance(info, dict)
    rendered = senv.render()
    assert isinstance(rendered, str) and len(rendered) > 0
    senv.set_state(state)
    assert senv.step_count == state["step_count"]
    print("  Stag Hunt API: PASS")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    test_api_smoke()
    test_harvest_greedy_vs_cooperative()
    test_staghunt_random_vs_coordinated()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
