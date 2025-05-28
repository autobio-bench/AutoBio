from pathlib import Path
import subprocess
import json
import textwrap
import shutil
import tempfile

def make_blender_script(camera: str, device_index: int) -> str:
    script = f"""
    import bpy

    def render_init(scene: bpy.types.Scene):
        cycles_preferences = bpy.context.preferences.addons['cycles'].preferences
        assert cycles_preferences['compute_device_type'] == 3, 'GPU compute device type is not set to OPTIX'
        cycles_devices = cycles_preferences['devices']
        for device in cycles_devices: device['use'] = 0
        cycles_devices[{device_index}]['use'] = 1

    bpy.app.handlers.render_init.append(render_init)
    bpy.context.scene.camera = bpy.data.objects['{camera}']
    """
    return textwrap.dedent(script)


def render(scene_path: Path, output_path: Path, camera: str, engine: str, device_index: int):
    blender_script = make_blender_script(camera, device_index)
    args = [
        "blender",
        "--background",
        str(scene_path),
        "--python-expr", blender_script,
        "--python-exit-code", "1",
        "--render-output", str(output_path),
        "--engine", engine,
        "--render-anim",
        "--render-format", "PNG",
    ]
    subprocess.run(args, check=True)

def ensure_scene(logdir: Path, fps: int, force: bool = False) -> Path:
    scene_path = logdir / "scene.blend"
    if scene_path.exists() and not force:
        return scene_path
    args = [
        "blender",
        "--background",
        "--python", "render_blender.py",
        "--",
        str(logdir),
        "--fps", str(fps),
    ]
    subprocess.run(args, check=True)
    return scene_path

def render_video(image_dir: Path, output_path: Path, fps: int):
    args = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", str(image_dir / "%04d.png"),
        "-filter:v", "format=rgba,premultiply=inplace=1",
        "-c:v", "libx264",
        "-profile:v", "high",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    subprocess.run(args, check=True)

def main(logdir: Path, engine: str, fps: int, device_index: int, force: bool):
    scene_path = ensure_scene(logdir, fps=fps, force=force)
    with open(logdir / "info.json", "r") as f:
        info = json.load(f)
    camera_mapping: dict[str, str] = info["task"]["camera_mapping"]
    for place, name in camera_mapping.items():
        # image_dir = logdir / "blender" / place
        # image_dir.mkdir(parents=True, exist_ok=True)
        # render(scene_path, image_dir / "####.png", name, engine, device_index)
        # render_video(image_dir, logdir / "blender" / f"{place}.mp4", fps=fps)
        # shutil.rmtree(image_dir)

        # with tempfile.TemporaryDirectory() as tmpdir:
        #     tmpdir = Path(tmpdir)
        #     render(scene_path, tmpdir / "####.png", name, engine, device_index)
        #     render_video(tmpdir, logdir / f"{place}-blender-2.mp4", fps=fps)
        render(scene_path, logdir / "blender" / place / "####.png", name, engine, device_index)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Render a Blender scene with a specific camera.")
    parser.add_argument("logdir", type=Path, help="Directory containing the scene file.")
    parser.add_argument("--engine", type=str, default="CYCLES", help="Rendering engine to use.")
    parser.add_argument("--fps", type=int, default=50, help="Frames per second for the output video.")
    parser.add_argument("--device_index", type=int, default=0, help="Index of the GPU device to use for rendering.")
    parser.add_argument("--force", action="store_true", help="Force recreation of the scene.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args.logdir, args.engine, args.fps, args.device_index, args.force)
