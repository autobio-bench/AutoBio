import os; os.environ['MUJOCO_GL'] = 'egl'
from pathlib import Path
import pickle

import mujoco
from mujoco.renderer import Renderer
import numpy as np
import imageio
from tqdm import trange

from serialize import load_log, take_state_split

def make_safe_name(name: str) -> str:
    return name.replace('/', '-').replace(':', '-')

def render_mujoco(
    save_dir: Path, model: mujoco.MjModel, qpos: np.ndarray,
    *,
    cameras: list[str] | None = None,
    height: int = 960,
    width: int = 1280,
    name_template: str = "{camera}.mp4",
    fps: int = 20,
    liquids: list[dict] | None = None,
    ui_state: list[dict] | None = None,
):
    save_dir = Path(save_dir)
    if cameras is None:
        camera_ids = list(range(model.ncam))
        cameras = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id) for camera_id in camera_ids]
        assert all(camera_name is not None for camera_name in cameras), f"Invalid camera id: {camera_ids}"
    else:
        camera_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) for name in cameras]
        assert all(camera_id != -1 for camera_id in camera_ids), f"Invalid camera name: {cameras}"
    camera_objects = []
    for camera_id in camera_ids:
        camera = mujoco.MjvCamera()
        camera.fixedcamid = camera_id
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera_objects.append(camera)

    if height > model.vis.global_.offheight:
        print(f"Warning: Requested height {height} is larger than model height {model.vis.global_.offheight}")
        model.vis.global_.offheight = height
    if width > model.vis.global_.offwidth:
        print(f"Warning: Requested width {width} is larger than model width {model.vis.global_.offwidth}")
        model.vis.global_.offwidth = width

    assert qpos.ndim == 2 and qpos.shape[1] == model.nq, f"Invalid qpos shape: {qpos.shape}"
    timestep = model.opt.timestep
    num_steps = qpos.shape[0]
    step = 1 / fps / timestep
    if not np.isclose(step, int(step)):
        print(f"Warning: Inexact step size {step} for timestep {timestep} and fps {fps}")
    indices = np.arange(0, num_steps, step)
    indices = np.rint(indices).astype(int)
    qpos = qpos[indices]
    num_frames = len(indices)

    if liquids is not None:
        from meshplane import Mesh, MeshPlane, make_plane_frame
        from skimage.measure import EllipseModel
        meshplanes = []
        for liquid in liquids:
            mesh = Mesh(liquid['vertices'], liquid['faces'].astype(np.uint64), liquid['boundary'])
            meshplane = MeshPlane(mesh)
            meshplanes.append(meshplane)
        def render_liquid(data, scene, render_context, i):
            for j in range(len(meshplanes)):
                meshplane = meshplanes[j]
                liquid = liquids[j]
                surface_normal = liquid['normal'][indices[i]]
                surface_distance = liquid['distance'][indices[i]]
                position = data.geom_xpos[liquid['geom_id']]
                rotation_matrix = data.geom_xmat[liquid['geom_id']].reshape(3, 3)

                meshplane.set_plane_normal(*surface_normal)

                def compose(local_pos, local_mat, global_pos=None, global_mat=None):
                    if global_mat is not None:
                        local_mat = global_mat @ local_mat
                        local_pos = global_mat @ local_pos
                    if global_pos is not None:
                        local_pos = global_pos + local_pos
                    return local_pos, local_mat

                # surface = meshplane.calculate_plane(surface_distance)
                # local_center = surface.center
                # local_frame = surface.frame
                # half_width = surface.half_width
                # half_height = surface.half_height
                # world_center, world_frame = compose(local_center, local_frame, position, rotation_matrix)
                # mujoco.mjv_initGeom(
                #     scene.geoms[scene.ngeom],
                #     type=mujoco.mjtGeom.mjGEOM_PLANE,
                #     size=(half_width, half_height, 1),
                #     pos=world_center,
                #     mat=world_frame.ravel(),
                #     rgba=(0, 0, 1, 1),
                # )
                liquid_mesh = meshplane.calculate_mesh(surface_distance)
                surface = liquid_mesh.vertices[liquid_mesh.boundary]
                local_frame = make_plane_frame(surface_normal)
                planar_surface = surface @ local_frame  # Make 2d
                em = EllipseModel()
                em.estimate(planar_surface[:, :2])
                xc, yc, a, b, theta = em.params
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                circle_pos = np.array([xc, yc, surface_distance])
                circle_mat = np.array([
                    [cos_theta * a, -sin_theta * b, 0.0],
                    [sin_theta * a,  cos_theta * b, 0.0],
                    [0.0, 0.0, 1.0],
                ])  # Transform unit circle to ellipse
                circle_pos, circle_mat = compose(circle_pos, circle_mat, None, local_frame)
                circle_pos, circle_mat = compose(circle_pos, circle_mat, position, rotation_matrix)
                circle_size = np.array([1.0, 1e-4, 0.0])
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=circle_size,
                    pos=circle_pos,
                    mat=circle_mat.ravel(),
                    rgba=(0, 0, 1, 1),
                )
                scene.ngeom += 1
    else:
        def render_liquid(data, scene, render_context, i):
            pass

    if ui_state is not None:
        figs = [
            state['trajectory'][0].make_canvas()
            for state in ui_state
        ]

        ui_current = [None] * len(ui_state)
        def render_ui(data, scene, render_context, i):
            for j in range(len(ui_state)):
                state = ui_state[j]
                ui_new = state['trajectory'][indices[i]]
                if ui_current[j] == ui_new:
                    continue
                ui_current[j] = ui_new
                
                texture_target = state['target']
                fig, ax = figs[j]
                ui_new.draw(ax)
                
                img = ui_new.render_canvas(fig)

                model.tex(texture_target).data[...] = img

                mujoco.mjr_uploadTexture(model, render_context, texture_target)
    else:
        def render_ui(data, scene, render_context, i):
            pass

    def render_custom(data, scene, render_context, i):
        render_liquid(data, scene, render_context, i)
        render_ui(data, scene, render_context, i)

    print(f"Rendering {num_frames} frames for cameras {cameras} to {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    camera_filenames = {
        camera: name_template.format(camera=make_safe_name(camera))
        for camera in cameras
    }

    downsample = {
        "fps": fps,
        "height": height,
        "width": width,
        "indices": indices.tolist(),
        "cameras": camera_filenames,
    }
    with open(save_dir / 'downsample.json', 'w') as f:
        import json
        json.dump(downsample, f, indent=4)

    renderer = Renderer(model, height, width)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    renderer._scene_option.sitegroup[:] = False  # Disable ALL sites rendering

    writers = [
        imageio.get_writer(save_dir / camera_filenames[camera], format='mp4', mode='I', fps=fps, codec='libx264', ffmpeg_params=['-crf', '18'])
        for camera in cameras
    ]

    data = mujoco.MjData(model)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in trange(num_frames, desc="Rendering"):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

        renderer.update_scene(data, camera_objects[0])
        render_custom(data, renderer.scene, renderer._mjr_context, i)

        for camera, writer in zip(camera_objects, writers):
            # Avoid updating scene repeatedly, only update camera
            mujoco.mjv_updateCamera(model, data, camera, renderer._scene)
            writer.append_data(renderer.render(out=image))

    for writer in writers:
        writer.close()
    renderer.close()

def render_mujoco_from_log(
    save_dir: Path, log_dir: Path | None = None,
    *,
    cameras: list[str] | None = None,
    height: int = 960,
    width: int = 1280,
    name_template: str = "{camera}.mp4",
    fps: int = 20,
    liquid: bool = False,
    ui_state: bool = False,
):
    if log_dir is None:
        log_dir = save_dir

    model, states, info = load_log(log_dir)
    qpos = take_state_split(states, info['split']['qpos'])

    if liquid:
        with open(log_dir / 'liquid.pkl', 'rb') as f:
            liquids = pickle.load(f)
    else:
        liquids = None

    if ui_state:
        with open(log_dir / 'ui_state.pkl', 'rb') as f:
            ui_state = pickle.load(f)
    else:
        ui_state = None

    render_mujoco(save_dir, model, qpos, cameras=cameras, height=height, width=width, name_template=name_template, fps=fps, liquids=liquids, ui_state=ui_state)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("save_dir", type=Path, help="Directory to save rendered videos")
    parser.add_argument("--log_dir", type=Path, default=None, help="Directory of the log file (if different from save_dir)")
    parser.add_argument("--cameras", nargs='*', default=None, help="List of camera names to render")
    parser.add_argument("--height", type=int, default=960, help="Height of the rendered video")
    parser.add_argument("--width", type=int, default=1280, help="Width of the rendered video")
    parser.add_argument("--name_template", type=str, default="{camera}.mp4", help="Template for output video filenames")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the rendered video")
    parser.add_argument("--liquid", action='store_true', help="Add liquid decoration to the rendered video")
    parser.add_argument("--ui", action='store_true', help="Render UI elements")

    args = parser.parse_args()

    mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

    render_mujoco_from_log(
        args.save_dir,
        log_dir=args.log_dir,
        cameras=args.cameras,
        height=args.height,
        width=args.width,
        name_template=args.name_template,
        fps=args.fps,
        liquid=args.liquid,
        ui_state=args.ui,
    )
