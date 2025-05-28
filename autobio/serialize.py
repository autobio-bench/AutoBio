import math
import json
import time
import os
from pathlib import Path
from io import BytesIO
from contextlib import contextmanager

import mujoco
import numpy as np
import zstandard as zstd
from tqdm import trange

STATES = {
    # Physics state
    "time": [mujoco.mjtState.mjSTATE_TIME, np.float64],
    "qpos": [mujoco.mjtState.mjSTATE_QPOS, np.float64, "nq"],
    "qvel": [mujoco.mjtState.mjSTATE_QVEL, np.float64, "nv"],
    "act": [mujoco.mjtState.mjSTATE_ACT, np.float64, "na"],
    "qacc_warmstart": [mujoco.mjtState.mjSTATE_WARMSTART, np.float64, "nv"],
    # Control (user) state
    "ctrl": [mujoco.mjtState.mjSTATE_CTRL, np.float64, "nu"],
    "qfrc_applied": [mujoco.mjtState.mjSTATE_QFRC_APPLIED, np.float64, "nv"],
    "xfrc_applied": [mujoco.mjtState.mjSTATE_XFRC_APPLIED, np.float64, "nbody", 6],
    "eq_active": [mujoco.mjtState.mjSTATE_EQ_ACTIVE, np.uint8, "neq"],
    "mocap_pos": [mujoco.mjtState.mjSTATE_MOCAP_POS, np.float64, "nmocap", 3],
    "mocap_quat": [mujoco.mjtState.mjSTATE_MOCAP_QUAT, np.float64, "nmocap", 4],
    "userdata": [mujoco.mjtState.mjSTATE_USERDATA, np.float64, "nuserdata"],
    "plugin_state": [mujoco.mjtState.mjSTATE_PLUGIN, np.float64, "npluginstate"],
}

def make_split(desc, model, offset):
    spec, dtype, *shape = desc
    resolved_shape = []
    for dim in shape:
        if isinstance(dim, str):
            dim = getattr(model, dim)
        resolved_shape.append(dim)
    size = mujoco.mj_stateSize(model, spec)
    assert math.prod(resolved_shape) == size
    next_offset = offset + size
    return {
        "spec": int(spec),
        "start": offset,
        "end": next_offset,
        "shape": resolved_shape,
        "dtype": np.dtype(dtype).str,
    }, next_offset

STATE_SPEC = sum(spec for spec, *_ in STATES.values())

class MujocoSerializer:
    def __init__(
        self,
        spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData,
        task: dict,
        log_root: Path,
        *,
        share_mjb: bool = True,
        initial_time: float = 0.0,
        log_name: str | None = None,
    ):
        assert data.time == initial_time, "Initial state mismatch"
        log_root = Path(log_root)
        if log_name is None:
            log_name = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.save_dir = log_root / log_name
        self.save_dir.mkdir(parents=True, exist_ok=False)

        self.model = model
        self.data = data
        self.task = task
        self.infos = []

        self.model_xml = spec.to_xml()

        model_size = mujoco.mj_sizeModel(model)
        model_bin = np.empty(model_size, dtype=np.uint8)
        mujoco.mj_saveModel(model, None, model_bin)
        if share_mjb:
            mjb_path = log_root / "model.mjb"
            if not mjb_path.exists():
                with open(mjb_path, "wb") as f:
                    model_bin.tofile(f)
            else:
                with open(mjb_path, "rb") as f:
                    existing = np.fromfile(f, dtype=np.uint8)
                assert np.array_equal(existing, model_bin), "Existing model.mjb differs. Model may have changed. Use a new root directory or set share_mjb=False."
        else:
            mjb_path = self.save_dir / "model.mjb"
            with open(mjb_path, "wb") as f:
                model_bin.tofile(f)
        
        self.last_time = data.time
        self.states = []
        self.state_size = mujoco.mj_stateSize(model, STATE_SPEC)
        self.state_split = {}
        offset = 0
        for name, value in STATES.items():
            desc, offset = make_split(value, model, offset)
            self.state_split[name] = desc
        assert offset == self.state_size

        self._record({})  # Record initial state

    def _record(self, info: dict):
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, self.data, state, STATE_SPEC)
        self.states.append(state)
        self.infos.append(info)

    def record(self, info: dict):
        time = self.data.time
        expected_time = self.last_time + self.model.opt.timestep
        assert math.isclose(time, expected_time), f"Expected time {expected_time}, got {time}"
        self._record(info)
        self.last_time = time

    def finish(self):
        self.states = np.stack(self.states)
        with open(self.save_dir / "model.xml", "w") as f:
            f.write(self.model_xml)
        with open(self.save_dir / "states.npy.zst", "wb") as f:
            with zstd.ZstdCompressor(level=3).stream_writer(f) as zstd_f:
                np.save(zstd_f, self.states)
        with open(self.save_dir / "info.json", "w") as f:
            json.dump({
                "task": self.task,
                "split": self.state_split,
                "info": self.infos,
            }, f, indent=2)
    
    @contextmanager
    def within_save_dir(self):
        old_cwd = os.getcwd()
        os.chdir(self.save_dir)
        try:
            yield
        finally:
            os.chdir(old_cwd)

    def validate(self):
        def take_split(arr, name):
            return take_state_split(arr, self.state_split[name])

        states = self.states
        model = self.model
        data = mujoco.MjData(model)

        def compare(index: int):
            state = states[index]
            for name in self.state_split:
                actual = getattr(data, name)
                expected = take_split(state, name)
                assert np.array_equal(actual, expected), f"{name}@{index}: {actual} != {expected}"
        
        data.qpos[:] = take_split(states[0], "qpos")
        data.qvel[:] = take_split(states[0], "qvel")
        data.act[:] = take_split(states[0], "act")

        for i in trange(1, len(states), desc="Validating"):
            data.ctrl[:] = take_split(states[i], 'ctrl')
            mujoco.mj_step(model, data)
            compare(i)

def take_state_split(arr, split):
    start = split['start']
    end = split['end']
    shape = tuple(split['shape'])
    dtype = split['dtype']
    return arr[..., start:end].reshape(arr.shape[:-1] + shape).astype(dtype)

def load_log(log_dir: Path):
    log_dir = Path(log_dir)
    model_path = log_dir / "model.mjb"
    if not model_path.exists():
        model_path = log_dir / ".." / "model.mjb"
    assert model_path.exists(), f"Model not found: {model_path}"

    model = mujoco.MjModel.from_binary_path(str(model_path))

    with open(log_dir / "states.npy.zst", "rb") as f:
        with zstd.ZstdDecompressor().stream_reader(f) as zstd_f:
            states_io = BytesIO(zstd_f.read())
    states = np.load(states_io)

    with open(log_dir / "info.json", "r") as f:
        info = json.load(f)

    return model, states, info
