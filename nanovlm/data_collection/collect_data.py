import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from minigrid.core.actions import Actions
from minigrid.wrappers import RGBImgPartialObsWrapper
from PIL import Image

from .env_utils import generate_state_description
from .config_loader import load_config, config_to_args

# Direction vectors for greedy navigation
_DIR_VECTORS = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def get_goal_pos(env: gym.Env) -> Tuple[int, int]:
    """Get goal position from environment"""
    unwrapped = env.unwrapped
    if hasattr(unwrapped, "goal_pos") and unwrapped.goal_pos is not None:
        return tuple(unwrapped.goal_pos)
    return (unwrapped.width - 2, unwrapped.height - 2)


def is_walkable(env: gym.Env, pos: Tuple[int, int]) -> bool:
    """Check if a position is walkable"""
    unwrapped = env.unwrapped
    x, y = pos
    if not (0 <= x < unwrapped.width and 0 <= y < unwrapped.height):
        return False
    obj = unwrapped.grid.get(*pos)
    return obj is None or obj.type in {"goal", "floor"}

def is_goal(env: gym.Env, pos: Tuple[int, int]) -> bool:
    """Check if a position is the goal"""
    #unwrapped = env.unwrapped
    goal_pos = get_goal_pos(env)
    return pos == goal_pos


def manhattan_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """Manhattan distance between two positions"""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def greedy_action(env: gym.Env, goal_pos: Tuple[int, int]) -> Optional[int]:
    """
    Compute greedy action for navigation:
    1. If moving forward reduces distance to goal and is walkable -> forward
    2. Otherwise -> turn toward goal with minimal rotation
    """
    unwrapped = env.unwrapped
    agent_pos = tuple(unwrapped.agent_pos)
    agent_dir = int(unwrapped.agent_dir)
    
    if agent_pos == goal_pos:
        return None  # Reached goal
    
    current_dist = manhattan_dist(agent_pos, goal_pos)
    
    # Try moving forward
    dx, dy = _DIR_VECTORS[agent_dir]
    next_pos = (agent_pos[0] + dx, agent_pos[1] + dy)
    
    if is_walkable(env, next_pos):
        next_dist = manhattan_dist(next_pos, goal_pos)
        if next_dist < current_dist:
            return int(Actions.forward)  # Forward reduces distance
    
    # Need to turn toward goal. Determine best direction.
    gx, gy = goal_pos
    ax, ay = agent_pos
    dx, dy = gx - ax, gy - ay
    
    # Pick goal direction (axis-aligned)
    if abs(dx) > abs(dy):
        goal_dir = 0 if dx > 0 else 2  # East or West
    else:
        goal_dir = 1 if dy > 0 else 3  # South or North
    
    # Compute minimal rotation
    diff = (goal_dir - agent_dir) % 4
    
    if diff == 0:
        return int(Actions.forward)
    elif diff == 1:
        return int(Actions.right)
    elif diff == 2:
        return int(Actions.right)  # Rotate either direction, choose right
    else:  # diff == 3
        return int(Actions.left)



class MiniGridExpertRunner:
    def __init__(
        self,
        env_id: str = "MiniGrid-Empty-8x8-v0",
        seed: int = 0,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        self.env_id = env_id
        self.seed = seed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.env = gym.make(env_id, render_mode=render_mode)

    def run_episode(
        self, seed: Optional[int] = None, save_frames: bool = False
    ) -> List[Dict[str, Any]]:
        obs, _ = self.env.reset(seed=seed if seed is not None else self.seed)
        traj: List[Dict[str, Any]] = []
        steps = 0
        
        goal_pos = get_goal_pos(self.env)

        while True:
            action = greedy_action(self.env, goal_pos)
            if action is None:
                # Already at goal, episode complete
                # Take "done" action to get goal reward
                next_obs, reward, terminated, truncated, step_info = self.env.step(int(Actions.done))
                frame = None
                if save_frames and self.render_mode == "rgb_array":
                    frame = self.env.render()
                traj.append(
                    {
                        "obs": obs,
                        "action": int(Actions.done),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info": step_info,
                        "frame": frame,
                    }
                )
                break

            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            frame = None
            if save_frames and self.render_mode == "rgb_array":
                frame = self.env.render()
            traj.append(
                {
                    "obs": obs,
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": step_info,
                    "frame": frame,
                }
            )

            obs = next_obs
            steps += 1
            if self.max_steps is not None and steps >= self.max_steps:
                break
            if terminated or truncated:
                break

        return traj

    def close(self) -> None:
        self.env.close()


def collect_data(
    env_ids: List[str],
    num_episodes: int,
    output_dir: Path,
    seed: int = 0,
    max_steps: Optional[int] = None,
    pomdp: bool = False,
) -> List[Dict[str, object]]:
    """Collect expert trajectories from one or more MiniGrid EmptyEnv sizes.

    Each record stores the action name in ``target`` and a state-aware text
    description in ``description``.

    Args:
        env_ids: List of Gymnasium env IDs to collect from.
        num_episodes: Episodes to collect *per env*.
        output_dir: Root directory for images and dataset.jsonl.
        seed: Base random seed; each episode offsets by episode index.
        max_steps: Cap steps per episode (None = unlimited).
        pomdp: If True, save partial observations (POMDP); if False, save full image (MDP).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, object]] = []

    prompt = "What action should the agent take to reach the goal?"
    
    obs_type = "pomdp" if pomdp else "mdp"

    action_names = {
        0: "turn_left",
        1: "turn_right",
        2: "forward",
        3: "pickup",
        4: "drop",
        5: "toggle",
        6: "done",
    }

    for env_id in env_ids:
        base_env = gym.make(env_id, render_mode="rgb_array")
        # Conditionally wrap environment based on observation type
        env = RGBImgPartialObsWrapper(base_env, tile_size=8) if pomdp else base_env
        # Sanitise env_id to a safe filename prefix (e.g. minigrid_empty_8x8_v0)
        env_tag = env_id.lower().replace("/", "_").replace("-", "_")

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            steps = 0
            episode_data: List[Dict[str, Any]] = []
            #rewards: List[float] = []
            goal_pos = get_goal_pos(base_env)

            while True:
                # Use greedy navigation algorithm
                action = greedy_action(base_env, goal_pos)
                
                if pomdp:
                    frame = obs["image"]
                else:
                    frame = base_env.render()
                image_path = images_dir / f"{env_tag}_{obs_type}_ep{ep}_step{steps}.png"
                Image.fromarray(frame).save(image_path)

                action_name = action_names[int(action)]
                description = generate_state_description(base_env)

                episode_data.append(
                    {
                        "image": str(image_path),
                        "prompt": prompt,
                        "target": action_name,
                        "description": description,
                        "action": int(action),
                        "env_id": env_id,
                        "episode": ep,
                        "step": steps,
                    }
                )

                obs, reward, terminated, truncated, _ = env.step(action)
                episode_data[-1]['reward'] = reward 
                episode_data[-1]['truncated'] = truncated
                episode_data[-1]['terminated'] = terminated
                steps += 1
                # if is_goal(env, tuple(base_env.unwrapped.agent_pos )):
                #     print(f"Reached goal at step {steps}. Taking 'done' action to complete episode."   )
                #     if pomdp:
                #         frame = obs["image"]
                #     else:
                #         frame = base_env.render()
                #     image_path = images_dir / f"{env_tag}_{obs_type}_ep{ep}_step{steps}.png"
                #     Image.fromarray(frame).save(image_path)

                #     action_name = action_names[int(action)]
                #     description = generate_state_description(base_env)
                #     episode_data.append(
                #         {
                #             "image": str(image_path),
                #             "prompt": prompt,
                #             "target": "done",
                #             "description": description,
                #             "action": int(Actions.done),
                #             "env_id": env_id,
                #             "episode": ep,
                #             "step": steps,
                #             'truncated': truncated,
                #             'terminated': terminated,
                #             'reward': 0,
                #         }
                #     )
                if max_steps is not None and steps >= max_steps:
                    break
                if terminated or truncated:
                    break

            # Calculate return-to-go (cumulative reward from current step onward)
            # for i, data in enumerate(episode_data):
            #     return_to_go = sum(rewards[i:])
            #     data["reward"] = rewards[i]
            #     data["return_to_go"] = return_to_go
            #     examples.append(data)
            examples.extend(episode_data)

        env.close()

    jsonl_path = output_dir / "dataset.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in examples:
            handle.write(json.dumps(row) + "\n")

    return examples


def collect_episodes(
    env_id: str,
    num_episodes: int,
    seed: int = 0,
    max_steps: Optional[int] = None,
    render_mode: Optional[str] = None,
    save_frames: bool = False,
) -> List[List[Dict[str, Any]]]:
    runner = MiniGridExpertRunner(
        env_id=env_id,
        seed=seed,
        max_steps=max_steps,
        render_mode=render_mode,
    )
    episodes: List[List[Dict[str, Any]]] = []

    for idx in range(num_episodes):
        episodes.append(runner.run_episode(seed=seed + idx, save_frames=save_frames))

    runner.close()
    return episodes


def _episode_success(traj: List[Dict[str, Any]]) -> bool:
    if not traj:
        return False
    return traj[-1].get("reward", 0.0) > 0.0


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(val) for val in value]
    return value


def save_trajectories_jsonl(episodes: List[List[Dict[str, Any]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(_jsonify(episode)) + "\n")


def save_trajectories_npz(episodes: List[List[Dict[str, Any]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, episodes=np.array(episodes, dtype=object))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect MiniGrid EmptyEnv data for SFT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use config.yaml in module)",
    )
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large", "dev", "random", "curriculum"],
        default=None,
        help="Use preset config (overrides config file settings)",
    )
    parser.add_argument(
        "--env-ids",
        nargs="+",
        default=None,
        metavar="ENV_ID",
        help="One or more env IDs to collect from, e.g. MiniGrid-Empty-5x5-v0 MiniGrid-Empty-8x8-v0",
    )
    parser.add_argument("--episodes", type=int, default=None, help="Override num_episodes (per env)")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
    parser.add_argument(
        "--pomdp",
        action="store_true",
        default=None,
        help="Save POMDP partial observations (default: MDP full images)",
    )
    args = parser.parse_args()

    # Load config from YAML
    config = load_config(config_path=args.config, preset=args.preset)
    
    # Convert to function arguments
    kwargs = config_to_args(config)
    
    # Override with CLI arguments if provided
    if args.env_ids:
        kwargs["env_ids"] = args.env_ids
    if args.episodes:
        kwargs["num_episodes"] = args.episodes
    if args.seed is not None:
        kwargs["seed"] = args.seed
    if args.max_steps is not None:
        kwargs["max_steps"] = args.max_steps
    if args.out_dir:
        kwargs["output_dir"] = Path(args.out_dir)
    if args.pomdp is not None:
        kwargs["pomdp"] = args.pomdp

    collect_data(**kwargs)
    print(f"Saved {len(kwargs.get('env_ids', []))} env(s), dataset at {kwargs['output_dir']}")


if __name__ == "__main__":
    main()
