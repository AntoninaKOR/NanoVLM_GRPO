from typing import Iterable, Optional, Tuple

import gymnasium as gym
from minigrid.core.actions import Actions


def get_goal_pos(env: gym.Env) -> Tuple[int, int]:
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "goal_pos") and unwrapped.goal_pos is not None:
        goal_pos = unwrapped.goal_pos
        return int(goal_pos[0]), int(goal_pos[1])
    return int(unwrapped.width - 2), int(unwrapped.height - 2)


def is_walkable(env: gym.Env, pos: Tuple[int, int]) -> bool:
    unwrapped = env.unwrapped
    x, y = pos
    if not (0 <= x < unwrapped.width and 0 <= y < unwrapped.height):
        return False
    obj = unwrapped.grid.get(*pos)
    if obj is None:
        return True
    return obj.type in {"goal", "floor"}


def neighbors(env: gym.Env, pos: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    x, y = pos
    candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
    for nx, ny in candidates:
        if is_walkable(env, (nx, ny)):
            yield (nx, ny)


def action_to_next(env: gym.Env, next_pos: Tuple[int, int]) -> Optional[int]:
    unwrapped = env.unwrapped
    agent_pos = tuple(unwrapped.agent_pos)
    agent_dir = int(unwrapped.agent_dir)

    if agent_pos == next_pos:
        return None

    dx = next_pos[0] - agent_pos[0]
    dy = next_pos[1] - agent_pos[1]

    if dx == 1:
        desired_dir = 0
    elif dx == -1:
        desired_dir = 2
    elif dy == 1:
        desired_dir = 1
    else:
        desired_dir = 3

    if agent_dir == desired_dir:
        return int(Actions.forward)

    diff = (desired_dir - agent_dir) % 4
    if diff in (1, 2):
        return int(Actions.right)
    return int(Actions.left)
