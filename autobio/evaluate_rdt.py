from collections import defaultdict
from pathlib import Path
import random
import subprocess
import sys
import time

from autobio_inference.websocket_client_policy import switch_policy

def evaluate_rdt(rdt_root: Path, save_dir: Path, host: str, port: int, num_episodes: int, image_history: int, num_workers: int, render_device_id: str, dry_run: bool):
    checkpoint_root = rdt_root / "checkpoints"

    task_seeds = defaultdict(lambda: random.randint(0, 2**32 - 1))

    for experiment_dir in checkpoint_root.iterdir():
        if not experiment_dir.is_dir():
            continue
        experiment_name = experiment_dir.name
        task_name = experiment_name.split("-")[0]
        if task_name == "thermal_cycler_combined":
            # thermal_cycler_combined is separately evaluated
            continue

        task_seed = task_seeds[task_name]

        save_file = save_dir / f"{experiment_name}.json"
        if save_file.exists():
            print(f"Skipping {experiment_name} as {save_file} already exists.")
            continue

        switch_args = [str(experiment_dir.relative_to(rdt_root))]
        evaluate_args = [
            sys.executable, "evaluate.py",
            "--host", host,
            "--port", str(port),
            "--task", task_name,
            "--num_episodes", str(num_episodes),
            "--image_history", str(image_history),
            "--num_workers", str(num_workers),
            "--save", str(save_file),
            "--seed", str(task_seed),  # ensure same seed for each task
            "--render_device_id", render_device_id,
        ]

        print(f"Would switch policy to {task_name} at {experiment_dir}")
        print(f"Would switch with command: {switch_args}")
        print(f"Would evaluate with command: {evaluate_args}")

        if dry_run:
            continue

        switch_policy(switch_args, host=host, port=port)
        time.sleep(5)  # Give some time for the server to switch the policy
        subprocess.run(evaluate_args, check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate rdt checkpoints using the WebSocket client.")
    parser.add_argument("--rdt_root", type=Path, required=True)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--image_history", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--render_device_id", type=str, default='0', help="Comma-separated list of GPU device IDs for rendering")
    parser.add_argument("--dry_run", action="store_true", help="If set, only print the commands without executing them.")
    args = parser.parse_args()
    evaluate_rdt(
        args.rdt_root,
        args.save_dir,
        args.host,
        args.port,
        args.num_episodes,
        args.image_history,
        args.num_workers,
        args.render_device_id,
        args.dry_run,
    )
