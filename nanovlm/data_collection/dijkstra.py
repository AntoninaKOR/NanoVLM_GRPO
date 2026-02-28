import heapq
from typing import Dict, List, Tuple

import gymnasium as gym

from .env_utils import get_goal_pos, neighbors


class Dijkstra:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def shortest_path(self) -> List[Tuple[int, int]]:
        unwrapped = self.env.unwrapped
        start = tuple(unwrapped.agent_pos)
        goal = get_goal_pos(self.env)

        if start == goal:
            return [start]

        frontier: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
        dist = {start: 0}
        prev: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while frontier:
            cost, current = heapq.heappop(frontier)
            if current == goal:
                break
            if cost != dist.get(current, 0):
                continue
            for nxt in neighbors(self.env, current):
                new_cost = cost + 1
                if new_cost < dist.get(nxt, 1_000_000):
                    dist[nxt] = new_cost
                    prev[nxt] = current
                    heapq.heappush(frontier, (new_cost, nxt))

        if goal not in prev and start != goal:
            return []

        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path
