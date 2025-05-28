import pickle
from pathlib import Path
import subprocess
import tempfile
import os

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from serialize import load_log, take_state_split

def render_ui(log_dir: Path, fps: int = 20):
    log_dir = Path(log_dir)
    model, states, info = load_log(log_dir)
    with open(log_dir / 'ui_state.pkl', 'rb') as f:
        ui_state = pickle.load(f)

    timestep = model.opt.timestep
    num_steps = states.shape[0]
    step = 1 / fps / timestep
    if not np.isclose(step, int(step)):
        print(f"Warning: Inexact step size {step} for timestep {timestep} and fps {fps}")
    indices = np.arange(0, num_steps, step)
    indices = np.rint(indices).astype(int)

    for state in ui_state:
        trajectory = state['trajectory']
        texture_target = state['target']
        fig, ax = trajectory[0].make_canvas()
        ui_current = None

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for i in range(len(indices)):
                ui_new = trajectory[indices[i]]
                if ui_current != ui_new:
                    ui_current = ui_new
                    ui_new.draw(ax)
                    fig.savefig(tmpdir / f"{i}.png", format='png', dpi=72 * 8)
                else:
                    os.link(tmpdir / f"{i-1}.png", tmpdir / f"{i}.png")
            plt.close(fig)
            subprocess.run([
                'ffmpeg',
                '-y',
                '-framerate', str(fps),
                '-i', str(tmpdir / '%d.png'),
                '-c:v', 'ffv1',
                str(log_dir / f"{texture_target}.mkv"),
            ], check=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path, help="Directory of the log file")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    args = parser.parse_args()

    mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
    render_ui(args.log_dir, fps=args.fps)
