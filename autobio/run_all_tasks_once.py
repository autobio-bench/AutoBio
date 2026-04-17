#!/usr/bin/env python3
"""
在 autobio 目录下执行：每条 benchmark 任务各跑 1 条轨迹，写入 log_all/<task>/0/，再调用 render.py 渲视频。

用法:
  cd AutoBio/autobio && python run_all_tasks_once.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_ALL = ROOT / "log_all"
RUN_NAME = "0"


def _chdir_autobio() -> None:
    os.chdir(ROOT)


def _load_plugin() -> None:
    import mujoco

    lib = ROOT / "libmjlab.so.3.3.0"
    if not lib.is_file():
        raise FileNotFoundError(f"缺少 MuJoCo 插件: {lib}（请在 autobio 目录下运行）")
    mujoco.mj_loadPluginLibrary(str(lib))


def make_expert(task_id: str):
    """构造对应任务的 Expert，并设置 expert.task（与 evaluate 中名称一致）。"""
    if task_id == "pickup":
        from pickup_centrifuge_tube import Pickup

        spec = Pickup.load()
        ex = Pickup.Expert(spec)
    elif task_id in ("thermal_cycler_close", "thermal_cycler_open"):
        from mani_thermal_cycler import ThermalCyclerManipulate

        spec = ThermalCyclerManipulate.load()
        ex = ThermalCyclerManipulate.Expert(spec)
    elif task_id == "insert":
        from transfer_centrifuge_tube import Insert

        spec = Insert.load()
        ex = Insert.Expert(spec)
    elif task_id == "pipette":
        from mani_pipette import Pipette

        spec = Pipette.load()
        ex = Pipette.Expert(spec)
    elif task_id == "screw_loose":
        from screw_loosen import ScrewLoose

        spec = ScrewLoose.load()
        ex = ScrewLoose.Expert(spec)
    elif task_id == "screw_tighten":
        from screw_tighten import ScrewTighten

        spec = ScrewTighten.load()
        ex = ScrewTighten.Expert(spec)
    elif task_id == "insert_centrifuge_5430":
        from load_centrifuge_5430 import InsertCentrifuge5430

        spec = InsertCentrifuge5430.load()
        ex = InsertCentrifuge5430.Expert(spec)
    elif task_id == "thermal_mixer":
        from mani_thermal_mixer import ThermalMixerManipulate

        spec = ThermalMixerManipulate.load()
        ex = ThermalMixerManipulate.Expert(spec)
    elif task_id == "centrifuge_5430_close_lid":
        from mani_centrifuge_5430 import Centrifuge5430Manipulate

        spec = Centrifuge5430Manipulate.load()
        ex = Centrifuge5430Manipulate.Expert(spec)
    elif task_id == "centrifuge_5910_lid_close":
        from mani_centrifuge_5910 import Centrifuge5910Manipulate

        spec = Centrifuge5910Manipulate.load()
        ex = Centrifuge5910Manipulate.Expert(spec)
    elif task_id == "centrifuge_mini_close_lid":
        from mani_centrifuge_mini import CentrifugeMiniManipulate

        spec = CentrifugeMiniManipulate.load()
        ex = CentrifugeMiniManipulate.Expert(spec)
    elif task_id == "vortex_mixer":
        from mani_vortex_mixer import VortexMixerManipulate

        spec = VortexMixerManipulate.load()
        ex = VortexMixerManipulate.Expert(spec)
    else:
        raise ValueError(f"未知任务: {task_id}")

    ex.task = task_id
    return ex


BENCHMARK_TASK_IDS: tuple[str, ...] = (
    "pickup",
    "thermal_cycler_close",
    "thermal_cycler_open",
    "insert",
    "pipette",
    "screw_loose",
    "screw_tighten",
    "insert_centrifuge_5430",
    "thermal_mixer",
    "centrifuge_5430_close_lid",
    "centrifuge_5910_lid_close",
    "centrifuge_mini_close_lid",
    "vortex_mixer",
)


def render_trajectory(traj_dir: Path, *, height: int = 224, width: int = 224, fps: int = 50) -> None:
    cmd = [
        sys.executable,
        str(ROOT / "render.py"),
        str(traj_dir),
        "--height",
        str(height),
        "--width",
        str(width),
        "--fps",
        str(fps),
    ]
    if (traj_dir / "liquid.pkl").exists():
        cmd.append("--liquid")
    if (traj_dir / "ui_state.pkl").exists():
        cmd.append("--ui")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> int:
    _chdir_autobio()
    _load_plugin()

    LOG_ALL.mkdir(parents=True, exist_ok=True)

    ok: list[str] = []
    failed: list[tuple[str, str]] = []

    for task_id in BENCHMARK_TASK_IDS:
        traj_dir = LOG_ALL / task_id / RUN_NAME
        print(f"\n=== [{task_id}] 采集 -> {traj_dir} ===", flush=True)
        try:
            expert = make_expert(task_id)
            expert.reset(0)
            expert.set_serializer(log_root=LOG_ALL / task_id, log_name=RUN_NAME)
            expert.execute()
        except Exception as e:
            failed.append((task_id, f"{e!r}\n{traceback.format_exc()}"))
            print(f"[{task_id}] 采集失败: {e}", flush=True)
            continue

        print(f"=== [{task_id}] 渲染 ===", flush=True)
        try:
            render_trajectory(traj_dir)
        except Exception as e:
            failed.append((task_id, f"render: {e!r}\n{traceback.format_exc()}"))
            print(f"[{task_id}] 渲染失败: {e}", flush=True)
            continue

        ok.append(task_id)

    print("\n======== 汇总 ========", flush=True)
    print(f"成功 ({len(ok)}): {', '.join(ok) or '(无)'}", flush=True)
    if failed:
        print(f"失败 ({len(failed)}):", flush=True)
        for name, msg in failed:
            print(f"  - {name}:\n{msg}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
