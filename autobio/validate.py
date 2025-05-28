import json
from pathlib import Path
from io import BytesIO

import mujoco
import numpy as np
import zstandard as zstd
from tqdm import tqdm

from serialize import take_state_split

def validate_one(model: mujoco.MjModel, states: np.ndarray, state_split: dict):
    def take_split(arr, name):
        return take_state_split(arr, state_split[name])

    data = mujoco.MjData(model)

    def compare(index: int):
        state = states[index]
        for name in state_split:
            actual = getattr(data, name)
            expected = take_split(state, name)
            assert np.array_equal(actual, expected), f"{name}@{index}: {actual} != {expected}"

    data.qpos[:] = take_split(states[0], "qpos")
    data.qvel[:] = take_split(states[0], "qvel")
    data.qacc_warmstart[:] = take_split(states[0], "qacc_warmstart")
    data.act[:] = take_split(states[0], "act")

    for i in range(1, len(states)):
        data.ctrl[:] = take_split(states[i], 'ctrl')
        mujoco.mj_step(model, data)
        data.eq_active[:] = take_split(states[i], 'eq_active')
        compare(i)

def validate(logroot: Path):
    model_path = logroot / "model.mjb"
    model = mujoco.MjModel.from_binary_path(str(model_path))

    logdirs = [
        log_dir for log_dir in logroot.iterdir()
        if log_dir.is_dir() and (log_dir / "states.npy.zst").exists()
    ]

    for log_dir in tqdm(logdirs, desc="Validating logs", unit="log"):
        with open(log_dir / "states.npy.zst", "rb") as f:
            with zstd.ZstdDecompressor().stream_reader(f) as zstd_f:
                states_io = BytesIO(zstd_f.read())
        states = np.load(states_io)

        with open(log_dir / "info.json", "r") as f:
            info = json.load(f)
        validate_one(model, states, info["split"])

if __name__ == "__main__":
    mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
    validate(Path("logs/insert_centrifuge_5430"))
