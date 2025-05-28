#!/usr/bin/env python

import json
import os
import numpy as np
from pathlib import Path
from io import BytesIO
import shutil

import zstandard as zstd
import cv2
import tyro

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
LEROBOT_HOME = Path(os.getenv("LEROBOT_HOME", "~/.cache/huggingface/lerobot")).expanduser()

def take_state_split(arr, split):
    start = split['start']
    end = split['end']
    shape = tuple(split['shape'])
    dtype = split['dtype']
    return arr[..., start:end].reshape(arr.shape[:-1] + shape).astype(dtype)

def load_log(log_dir: Path):
    log_dir = Path(log_dir)
    with open(log_dir / "states.npy.zst", "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as zstd_f:
            states_io = BytesIO(zstd_f.read())
    states = np.load(states_io)
    with open(log_dir / "info.json", "r") as f:
        info = json.load(f)
    with open(log_dir / "downsample.json", "r") as f:
        downsample = json.load(f)
    return states, info, downsample

def confirm(message: str):
    """Prompt the user for confirmation."""
    import sys
    if not sys.stdin.isatty():
        return True  # Assume yes if not in a terminal
    while True:
        response = input(f"{message} (y/n): ").strip().lower()
        if response in ("y", "yes"):
            return True
        elif response in ("n", "no"):
            return False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")

def probe_log(log_dir: Path):
    with open(log_dir / "info.json", "r") as f:
        info = json.load(f)
    task = info["task"]
    state_dim = len(task["state_indices"])
    action_dim = len(task["action_indices"])
    with open(log_dir / "downsample.json", "r") as f:
        downsample = json.load(f)
    fps = downsample["fps"]
    height = downsample["height"]
    width = downsample["width"]
    camera_mapping = task["camera_mapping"]
    return fps, height, width, state_dim, action_dim, list(camera_mapping.keys())

def main(data_dir: str, repo_id: str):
    parent_path = Path(data_dir)
    output_path = LEROBOT_HOME / repo_id

    if output_path.exists():
        if confirm(f"Output path {output_path} already exists. Delete it?"):
            shutil.rmtree(output_path)
        else:
            print("Exiting without changes.")
            return

    log_folders = sorted(d for d in parent_path.iterdir() if d.is_dir())
    assert len(log_folders) > 0, "No log folders found in the specified directory."

    fps, height, width, state_dim, action_dim, camera_keys = probe_log(log_folders[0])
    image_shape = (height, width, 3)
    print(f"FPS: {fps}")
    print(f"Image shape: {image_shape}")
    print(f"State Dimension: {state_dim}")
    print(f"Action Dimension: {action_dim}")
    print(f"Camera keys: {camera_keys}")

    features ={
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }
    for camera_key in camera_keys:
        features[camera_key] = {
            "dtype": "video",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
    )

    for log_dir in log_folders:
        states, info, downsample = load_log(log_dir)
        
        state_splits = info["split"]
        task = info["task"]
        indices = downsample["indices"]
        prompt = task["prefix"]
        state_indices = task["state_indices"]
        action_indices = task["action_indices"]
        assert len(state_indices) == state_dim
        assert len(action_indices) == action_dim
        assert downsample["fps"] == fps

        camera_mapping = task["camera_mapping"]
        camera_files = downsample["cameras"]
        def get_camera_file(camera_name):
            camera_file = camera_files[camera_name]
            camera_stream = cv2.VideoCapture(str(log_dir / camera_file))
            if not camera_stream.isOpened():
                raise RuntimeError(f"Failed to open video stream for {log_dir}")
            return camera_stream
        camera_streams = {camera_key: get_camera_file(camera) for camera_key, camera in camera_mapping.items()}

        for i in indices:
            state_record = states[i]
            qpos = take_state_split(state_record, state_splits["qpos"])
            ctrl = take_state_split(state_record, state_splits["ctrl"])

            frame = {
                "state": qpos[state_indices].astype(np.float32),
                "actions": ctrl[action_indices].astype(np.float32),
            }

            for camera_key, camera_stream in camera_streams.items():
                ret, image = camera_stream.read()
                assert ret, f"Failed to read image at index {i} from {log_dir}"
                assert image.shape == image_shape, f"Image shape mismatch at index {i}: {image.shape} != {image_shape}"
                frame[camera_key] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            dataset.add_frame(frame)

        for camera_stream in camera_streams.values():
            assert not camera_stream.grab(), f"Not all frames were read from {log_dir}"
            camera_stream.release()

        dataset.save_episode(task=prompt)

    dataset.consolidate(run_compute_stats=True)

if __name__ == "__main__":
    tyro.cli(main)
