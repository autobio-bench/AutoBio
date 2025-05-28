import os
from pathlib import Path

import numpy as np
import torch
import yaml
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata, MultiLeRobotDataset

from configs.state_vec import STATE_VEC_IDX_MAPPING


def load_dataset(
    repo_id: str,
    *,
    num_episodes: int | None = None,
    action_horizon: int = 50,
    image_history: int | None = None,
    local_files_only: bool = True,
    root: Path | None = None
) -> tuple[LeRobotDataset, dict[int, str]]:
    if num_episodes is not None:
        episodes = list(range(num_episodes))
    else:
        episodes = None
    if root is not None:
        root = Path(root) / repo_id

    dataset_meta = LeRobotDatasetMetadata(repo_id, root=root, local_files_only=local_files_only)
    delta_timestamps={
        "actions": [t / dataset_meta.fps for t in range(action_horizon)],
    }
    if image_history is not None:
        video_delta = [-t / dataset_meta.fps for t in range(image_history)]
        video_delta.reverse()
        for key in dataset_meta.video_keys:
            delta_timestamps[key] = video_delta
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        local_files_only=local_files_only,
    )

    video_keys = tuple(sorted(dataset.meta.video_keys))
    lang_embeds = np.load(dataset.root / "t5-v1_1-xxl.npz")
    text_embeds = lang_embeds["text_embeds"]
    attn_mask = lang_embeds["attn_mask"]
    stats = dataset.meta.stats
    robot_type = dataset.meta.robot_type
    tasks = dataset.meta.tasks

    return dataset, tasks, video_keys, text_embeds, attn_mask, stats, robot_type

def load_multi_dataset(
    repo_ids: list[str],
    *,
    action_horizon: int = 50,
    image_history: int | None = None,
    local_files_only: bool = True,
    root: Path,
) -> tuple[MultiLeRobotDataset, list[dict[int, str]]]:
    all_fps = []
    all_tasks = []
    all_video_keys = []
    all_robot_types = []
    text_embeds = []
    attn_mask = []
    for repo_id in repo_ids:
        dataset_meta = LeRobotDatasetMetadata(repo_id, root=root / repo_id, local_files_only=local_files_only)
        all_fps.append(dataset_meta.fps)
        all_tasks.append(dataset_meta.tasks)
        all_video_keys.append(tuple(sorted(dataset_meta.video_keys)))
        all_robot_types.append(dataset_meta.robot_type)

        lang_embeds = np.load(dataset_meta.root / "t5-v1_1-xxl.npz")
        text_embeds.append(lang_embeds["text_embeds"])
        attn_mask.append(lang_embeds["attn_mask"])

    assert len(set(all_fps)) == 1, "All datasets must have the same fps"
    assert len(set(all_video_keys)) == 1, "All datasets must have the same video keys"
    assert len(set(all_robot_types)) == 1, "All datasets must have the same robot type"
    fps = all_fps[0]
    video_keys = all_video_keys[0]
    robot_type = all_robot_types[0]
    delta_timestamps={
        "actions": [t / fps for t in range(action_horizon)],
    }
    if image_history is not None:
        video_delta = [-t / fps for t in range(image_history)]
        video_delta.reverse()
        for key in video_keys:
            delta_timestamps[key] = video_delta
    dataset = MultiLeRobotDataset(
        repo_ids,
        root=root,
        delta_timestamps=delta_timestamps,
        local_files_only=local_files_only,
    )
    stats = dataset.stats
    return dataset, all_tasks, video_keys, text_embeds, attn_mask, stats, robot_type

class LeRobotVLADataset:
    def __init__(self):
        assert "LEROBOT_TASK" in os.environ, "Set LEROBOT_TASK to the task name"
        assert "LEROBOT_ROOT" in os.environ, "Set LEROBOT_ROOT to the root of your datasets"
        num_episodes = int(os.environ.get("LEROBOT_NUM_EPISODES", 0))

        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']

        self.DATASET_NAME = os.environ["LEROBOT_TASK"]
        self.DATASET_ROOT = os.environ["LEROBOT_ROOT"]
        if "LEROBOT_MULTI_TASKS" in os.environ:
            names = os.environ["LEROBOT_MULTI_TASKS"].split(",")
            repo_ids = [f"data/{name}" for name in names]
            self.dataset, self.tasks, video_keys, self.text_embeds, self.attn_mask, stats, robot_type = load_multi_dataset(
                repo_ids,
                action_horizon=self.CHUNK_SIZE,
                image_history=self.IMG_HISORY_SIZE,
                local_files_only=True,
                root=Path(self.DATASET_ROOT)
            )
            self.multi_dataset = True
        else:
            self.dataset, self.tasks, video_keys, self.text_embeds, self.attn_mask, stats, robot_type = load_dataset(
                f"data/{self.DATASET_NAME}",
                num_episodes=num_episodes if num_episodes > 0 else None,
                action_horizon=self.CHUNK_SIZE,
                image_history=self.IMG_HISORY_SIZE,
                local_files_only=True,
                root=Path(self.DATASET_ROOT)
            )
            self.multi_dataset = False

        self.dualarm = "wrist_image_2" in video_keys
        UNI_STATE_INDICES = [STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)] + [STATE_VEC_IDX_MAPPING["right_gripper_open"]]
        if self.dualarm:
            UNI_STATE_INDICES += [STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)] + [STATE_VEC_IDX_MAPPING["left_gripper_open"]]
        def fill_in_state(values: np.ndarray):
            uni_vec = np.zeros(values.shape[:-1] + (self.STATE_DIM,))
            uni_vec[..., UNI_STATE_INDICES] = values
            return uni_vec
        self.state_stat = state_stat = stats["state"]

        state_min = state_stat["min"]
        state_max = state_stat["max"]
        self.state_min = torch.zeros_like(state_min, dtype=torch.float32)
        self.state_max = torch.ones_like(state_max, dtype=torch.float32)
        self.state_min[6] = state_min[6]
        self.state_max[6] = state_max[6]
        if self.dualarm:
            self.state_min[13] = state_min[13]
            self.state_max[13] = state_max[13]
        self.actions_min = self.state_min.clone()
        self.actions_max = self.state_max.clone()
        if robot_type == "ur5e":
            self.actions_min[6] = 0.0
            self.actions_max[6] = 255.0
            if self.dualarm:
                self.actions_min[13] = 0.0
                self.actions_max[13] = 255.0

        state_mean = (state_stat["mean"] - state_min) / (state_max - state_min)
        state_std = state_stat["std"] / (state_max - state_min)
        self.uni_state_mean = fill_in_state(state_mean)
        self.uni_state_std = fill_in_state(state_std)
        self.uni_state_indicator = fill_in_state(np.ones(len(state_mean)))
        self._fill_in_state = fill_in_state

    def __len__(self):
        return len(self.dataset)
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        if self.multi_dataset:
            dataset_index = sample["dataset_index"]
            text_embed = self.text_embeds[dataset_index][sample["task_index"]]
            attn_mask = self.attn_mask[dataset_index][sample["task_index"]]
        else:
            text_embed = self.text_embeds[sample["task_index"]]
            attn_mask = self.attn_mask[sample["task_index"]]
        meta = {
            "dataset_name": self.DATASET_NAME,
            # "#steps": num_steps,
            "step_id": sample["frame_index"],
            # "instruction": self.tasks[sample["task_index"]],
            "lang_embed": text_embed[attn_mask],
        }
        def convert_image(x: torch.Tensor):
            x = x.numpy()
            x = (x * 255).astype(np.uint8)
            x = x.transpose(0, 2, 3, 1)
            return x
        cam_high = convert_image(sample["image"])
        cam_high_mask = ~sample["image_is_pad"]
        cam_right_wrist = convert_image(sample["wrist_image"])
        cam_right_wrist_mask = ~sample["wrist_image_is_pad"]
        if self.dualarm:
            cam_left_wrist = convert_image(sample["wrist_image_2"])
            cam_left_wrist_mask = ~sample["wrist_image_2_is_pad"]
        else:
            cam_left_wrist = np.zeros((self.IMG_HISORY_SIZE, 0, 0, 0), dtype=np.uint8)
            cam_left_wrist_mask = np.zeros((self.IMG_HISORY_SIZE,), dtype=np.bool_)

        def _normalize(x, xmin, xmax):
            return (x - xmin) / (xmax - xmin)

        state = _normalize(sample["state"], self.state_min, self.state_max)
        actions = _normalize(sample["actions"], self.actions_min, self.actions_max)
        state = self._fill_in_state(state)[None]
        actions = self._fill_in_state(actions)

        res = {
            "meta": meta,
            "state": state,
            "state_std": self.uni_state_std,
            "state_mean": self.uni_state_mean,
            # "state_norm": state_norm,
            "actions": actions,
            "state_indicator": self.uni_state_indicator,
            "cam_high": cam_high,
            "cam_high_mask": cam_high_mask,
            "cam_left_wrist": cam_left_wrist,
            "cam_left_wrist_mask": cam_left_wrist_mask,
            "cam_right_wrist": cam_right_wrist,
            "cam_right_wrist_mask": cam_right_wrist_mask
        }
        res = {
            k: v.numpy() if isinstance(v, torch.Tensor) else v
            for k, v in res.items()
        }
        return res

    def save_normalization(self, path: str):
        with open(path, "w") as f:
            import json
            json.dump({
                "state_min": self.state_min.tolist(),
                "state_max": self.state_max.tolist(),
                "actions_min": self.actions_min.tolist(),
                "actions_max": self.actions_max.tolist(),
            }, f)
