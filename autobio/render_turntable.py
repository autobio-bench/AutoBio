"""
如何启动本脚本:

渲染一个Mujoco XML模型的转台(转圈)效果视频。

用法示例：
    python render_turntable.py --xml_path path/to/your_model.xml --output output/video.mp4

可选参数：
    --seconds       旋转一圈所用秒数（默认 8.0）
    --fps           帧率（默认 30）
    --width         输出视频宽度（默认 1280）
    --height        输出视频高度（默认 720）
    --elevation     相机仰角（默认 -20.0）
    --distance-scale 相机距离缩放因子（默认 1.8）

更多帮助：
    python render_turntable.py --help
"""

import argparse
from pathlib import Path

import imageio
import mujoco
import numpy as np
from mujoco.renderer import Renderer


def load_plugin_if_exists(base_dir: Path) -> None:
    plugin = base_dir / "libmjlab.so.3.3.0"
    if plugin.exists():
        mujoco.mj_loadPluginLibrary(str(plugin))


def render_turntable(
    xml_path: Path,
    output: Path,
    *,
    seconds: float = 8.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    elevation: float = -20.0,
    distance_scale: float = 1.8,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Ensure offscreen framebuffer is large enough for requested output size.
    if width > model.vis.global_.offwidth:
        model.vis.global_.offwidth = width
    if height > model.vis.global_.offheight:
        model.vis.global_.offheight = height

    renderer = Renderer(model, height=height, width=width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = model.stat.center
    cam.distance = max(model.stat.extent * distance_scale, 0.2)
    cam.elevation = elevation

    total_frames = max(int(seconds * fps), 1)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        output, format="mp4", mode="I", fps=fps, codec="libx264", ffmpeg_params=["-crf", "18"]
    )
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    try:
        for i in range(total_frames):
            cam.azimuth = 360.0 * i / total_frames
            renderer.update_scene(data, camera=cam)
            writer.append_data(renderer.render(out=frame))
    finally:
        writer.close()
        renderer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a static turntable video from an MJCF XML.")
    parser.add_argument("xml", type=Path, help="Path to XML scene/model file")
    parser.add_argument("--output", type=Path, required=True, help="Output mp4 path")
    parser.add_argument("--seconds", type=float, default=8.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--elevation", type=float, default=-20.0, help="Camera elevation angle")
    parser.add_argument(
        "--distance-scale",
        type=float,
        default=1.8,
        help="Distance multiplier based on model extent",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    load_plugin_if_exists(base_dir)
    render_turntable(
        args.xml,
        args.output,
        seconds=args.seconds,
        fps=args.fps,
        width=args.width,
        height=args.height,
        elevation=args.elevation,
        distance_scale=args.distance_scale,
    )


if __name__ == "__main__":
    main()
