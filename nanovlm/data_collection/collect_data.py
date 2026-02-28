import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from PIL import Image

from .dijkstra import Dijkstra
from .env_utils import action_to_next
from .config_loader import load_config, config_to_args


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
        path: List[Tuple[int, int]] = []
        path_index = 0

        while True:
            if not path or path_index >= len(path) - 1:
                path = Dijkstra(self.env).shortest_path()
                path_index = 0

            if not path:
                break

            next_pos = path[min(path_index + 1, len(path) - 1)]
            action = action_to_next(self.env, next_pos)
            if action is None:
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
            path_index = min(path_index + 1, len(path) - 1)
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
    mode: str = "action",
    max_steps: Optional[int] = None,
) -> List[Dict[str, object]]:
    """Collect expert trajectories from one or more MiniGrid EmptyEnv sizes.

    Images are saved at their native render resolution. Resizing to the model's
    expected input size is handled by the NanoVLM image processor at training time.

    Args:
        env_ids: List of Gymnasium env IDs to collect from (e.g.
            ["MiniGrid-Empty-5x5-v0", "MiniGrid-Empty-8x8-v0"]).
        num_episodes: Episodes to collect *per env*.
        output_dir: Root directory for images and dataset.jsonl.
        seed: Base random seed; each episode offsets by episode index.
        mode: "action" for action-name labels; "text_action" for description+action.
        max_steps: Cap steps per episode (None = unlimited).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, object]] = []

    prompt = (
        "What action should the agent take to reach the goal?"
        if mode == "action"
        else "Describe what you see and what action the agent should take."
    )

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
        env = gym.make(env_id, render_mode="rgb_array")
        planner = Dijkstra(env)
        # Sanitise env_id to a safe filename prefix (e.g. minigrid_empty_8x8_v0)
        env_tag = env_id.lower().replace("/", "_").replace("-", "_")

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)
            path: List[Tuple[int, int]] = []
            path_idx = 0
            steps = 0

            while True:
                if not path or path_idx >= len(path) - 1:
                    path = planner.shortest_path()
                    path_idx = 0
                if not path:
                    break

                next_pos = path[min(path_idx + 1, len(path) - 1)]
                action = action_to_next(env, next_pos)
                if action is None:
                    break

                # Save the raw rendered frame; the NanoVLM processor handles resizing.
                frame = env.render()
                image_path = images_dir / f"{env_tag}_ep{ep}_step{steps}.png"
                Image.fromarray(frame).save(image_path)

                action_name = action_names[int(action)]
                if mode == "action":
                    target = action_name
                else:
                    target = f"The agent needs to navigate to the goal. Action: {action_name}"

                examples.append(
                    {
                        "image": str(image_path),
                        "prompt": prompt,
                        "target": target,
                        "action": int(action),
                        "env_id": env_id,
                        "episode": ep,
                        "step": steps,
                    }
                )

                obs, _, terminated, truncated, _ = env.step(action)
                steps += 1
                path_idx = min(path_idx + 1, len(path) - 1)
                if max_steps is not None and steps >= max_steps:
                    break
                if terminated or truncated:
                    break

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
    parser.add_argument("--mode", choices=["action", "text_action"], default=None, help="Override mode")
    parser.add_argument("--out-dir", type=str, default=None, help="Override output directory")
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
    if args.mode:
        kwargs["mode"] = args.mode
    if args.out_dir:
        kwargs["output_dir"] = Path(args.out_dir)

    collect_data(**kwargs)
    print(f"Saved {len(kwargs.get('env_ids', []))} env(s), dataset at {kwargs['output_dir']}")


if __name__ == "__main__":
    main()
