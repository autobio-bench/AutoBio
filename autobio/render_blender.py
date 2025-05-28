from contextlib import contextmanager
import json
from pathlib import Path
from io import BytesIO
import re

import bpy
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco
import zstandard as zstd
import trimesh

mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

@contextmanager
def temporary_active_collection(collection: bpy.types.Collection):
    view_layer = bpy.context.view_layer
    previous = view_layer.active_layer_collection
    view_layer.active_layer_collection = view_layer.layer_collection.children[collection.name]
    try:
        yield
    finally:
        view_layer.active_layer_collection = previous

def add_mesh(name: str, vertices: np.ndarray, faces: np.ndarray) -> bpy.types.Mesh:
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices=vertices, edges=[], faces=faces, shade_flat=True)
    mesh.validate()
    mesh.materials.append(None)
    return mesh

def import_obj_mesh(path: Path):
    bpy.ops.wm.obj_import(
        filepath=str(path),
        forward_axis='Y',
        up_axis='Z',
        use_split_objects=False,
        use_split_groups=False,
    )

def import_stl_mesh(path: Path):
    bpy.ops.wm.stl_import(
        filepath=str(path),
        forward_axis='Y',
        up_axis='Z',
    )

def import_gltf_mesh(path: Path):
    bpy.ops.import_scene.gltf(
        filepath=str(path),
    )

def import_mesh(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".obj":
        import_obj_mesh(path)
    elif suffix == ".stl":
        import_stl_mesh(path)
    else:
        raise NotImplementedError

def srgb_to_linearrgb(c: float) -> float:
    if c < 0.04045:
        return 0.0 if c < 0.0 else c * (1.0 / 12.92)
    return pow((c + 0.055) * (1.0 / 1.055), 2.4)

def srgb_to_linear(c: tuple[float, ...]) -> tuple[float, ...]:
    assert len(c) == 3
    return tuple(srgb_to_linearrgb(x) for x in c)

def add_material(name: str, diffuse_srgb: np.ndarray) -> bpy.types.Material:
    # TODO: handle textures?
    alpha = diffuse_srgb[3]
    diffuse_srgb = diffuse_srgb[:3]
    diffuse_linear = srgb_to_linear(diffuse_srgb)
    material = bpy.data.materials.new(name)
    material.diffuse_color = (*diffuse_linear, alpha)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = (*diffuse_linear, 1)
    if alpha < 1:
        bsdf.inputs['Alpha'].default_value = alpha
    return material

# Currently assumes perspective camera without principal point offset
def add_fixed_camera(
    name: str,
    fovy: float,
    clip_start: float,
    clip_end: float,
) -> bpy.types.Camera:
    camera = bpy.data.cameras.new(name)
    camera.lens_unit = 'FOV'
    camera.sensor_fit = 'VERTICAL'
    camera.angle = fovy
    camera.clip_start = clip_start
    camera.clip_end = clip_end
    return camera

def add_object(
    name: str,
    data: bpy.types.Mesh | None = None,
    parent: bpy.types.Object | None = None,
    collection: bpy.types.Collection | None = None,
    location: np.ndarray | None = None,
    rotation_quaternion: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    material: bpy.types.Material | None = None,
    *,
    allow_rename: bool = False,
) -> bpy.types.Object:
    obj = bpy.data.objects.new(name, data)
    if not allow_rename and name != obj.name:
        raise ValueError(f"Invalid object name: {name} != {obj.name}")
    obj.rotation_mode = 'QUATERNION'
    if collection is None:
        collection = bpy.context.scene.collection
    collection.objects.link(obj)
    if parent is not None:
        obj.parent = parent
    if location is not None:
        obj.location = location
    if rotation_quaternion is not None:
        obj.rotation_quaternion = rotation_quaternion
    if scale is not None:
        obj.scale = scale
    if material is not None:
        obj.material_slots[0].link = 'OBJECT'
        obj.material_slots[0].material = material
    return obj

def copy_object_with_geom(
    obj: bpy.types.Object,
    prefix: str,
    parent: bpy.types.Object | None = None,
    collection: bpy.types.Collection | None = None,
    location: np.ndarray | None = None,
    rotation_quaternion: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    *,
    allow_rename: bool = False,
) -> bpy.types.Object:
    copy = obj.copy()
    name = prefix + obj.name
    copy.name = name
    if not allow_rename and name != copy.name:
        raise ValueError(f"Invalid object name: {name} != {copy.name}")
    if collection is None:
        collection = bpy.context.scene.collection
    collection.objects.link(copy)
    if parent is not None:
        copy.parent = parent
    if location is not None:
        copy.location = location
    if rotation_quaternion is not None:
        copy.rotation_mode = 'QUATERNION'
        copy.rotation_quaternion = rotation_quaternion
    if scale is not None:
        copy.scale = scale
    for geom in obj.children:
        if geom.type == 'MESH':
            assert len(geom.children) == 0
            copy_geom = geom.copy()
            copy_geom.name = prefix + geom.name
            collection.objects.link(copy_geom)
            copy_geom.parent = copy
    return copy

def ensure_collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection

def ensure_material(name: str, diffuse_srgb: np.ndarray) -> bpy.types.Material:
    material = bpy.data.materials.get(name)
    if material is None:
        material = add_material(name, diffuse_srgb)
    return material

class MeshManager:
    def __init__(self, model: mujoco.MjModel, mesh_dir: Path, scratch: bpy.types.Collection, primitive_resolution: int = 16):
        self.model = model
        self.mesh_dir = mesh_dir
        self.scratch = scratch
        self.primitive_resolution = primitive_resolution
        self.blender_plane = None
        self.blender_sphere = None
        self.blender_capsule = {}
        self.blender_cylinder = None
        self.blender_box = None
        self.blender_meshes = {}

    def add_plane(self, half_width: float, half_height: float):
        if self.blender_plane is None:
            vertices = np.array([
                [-1, -1, 0],
                [1, -1, 0],
                [1, 1, 0],
                [-1, 1, 0],
            ])
            faces = np.array([
                [0, 1, 2, 3],
            ])
            self.blender_plane = add_mesh("primitive.plane", vertices, faces)
        return self.blender_plane, np.array((half_width, half_height, 1))
    
    def add_sphere(self, radius: float):
        if self.blender_sphere is None:
            mesh = trimesh.creation.uv_sphere(count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_sphere = add_mesh("primitive.sphere", mesh.vertices, mesh.faces)
        return self.blender_sphere, np.array((radius, radius, radius))
    
    def add_capsule(self, radius: float, half_height: float):
        ratio = half_height / radius
        if ratio not in self.blender_capsule:
            mesh = trimesh.creation.capsule(radius=1, height=2 * ratio, count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_capsule[ratio] = add_mesh(f"primitive.capsule", mesh.vertices, mesh.faces)
        return self.blender_capsule[ratio], np.array((radius, radius, radius))
    
    def add_ellipsoid(self, radii: np.ndarray):
        if self.blender_sphere is None:
            mesh = trimesh.creation.uv_sphere(count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_sphere = add_mesh("primitive.sphere", mesh.vertices, mesh.faces)
        return self.blender_sphere, radii

    def add_cylinder(self, radius: float, half_height: float):
        if self.blender_cylinder is None:
            mesh = trimesh.creation.cylinder(radius=1, height=2, count=[self.primitive_resolution, self.primitive_resolution])
            self.blender_cylinder = add_mesh("primitive.cylinder", mesh.vertices, mesh.faces)
        return self.blender_cylinder, np.array((radius, radius, half_height))
    
    def add_box(self, half_extents: np.ndarray):
        if self.blender_box is None:
            mesh = trimesh.creation.box(extents=(2,) * 3)
            self.blender_box = add_mesh("primitive.box", mesh.vertices, mesh.faces)
        return self.blender_box, half_extents

    def add_mesh(self, id: int, prefix: str = ""):
        if id in self.blender_meshes:
            return self.blender_meshes[id]
        mesh = self.model.mesh(id)
        name = remove_prefix(mesh.name, prefix)
        try:
            ret = self._add_mesh_from_file(id, name)
        except FileNotFoundError:
            ret = self._add_mesh_from_data(id, mesh)
        self.blender_meshes[id] = ret
        return ret
    
    def _add_mesh_from_file(self, id: int, name: str):
        pathadr = self.model.mesh_pathadr[id].item()
        if pathadr == -1:
            raise FileNotFoundError
        paths = self.model.paths[pathadr:]
        null = paths.index(0)
        path = self.mesh_dir / paths[:null].decode('utf-8')

        assert len(self.scratch.objects) == 0
        with temporary_active_collection(self.scratch):
            import_mesh(path)
        
        meshes = []
        for obj in self.scratch.objects:
            if obj.type == 'MESH':
                meshes.append(obj.data)
        if len(meshes) != 1:
            raise ValueError(f"Expected 1 mesh, found {len(meshes)}")
        
        mesh = meshes[0]
        mesh.name = name
        assert mesh.name == name
        mesh.materials.append(None)

        pos = self.model.mesh_pos[id]
        quat = self.model.mesh_quat[id]
        scale = self.model.mesh_scale[id]

        for obj in self.scratch.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        return mesh, pos, quat, scale

    def _add_mesh_from_data(self, id: int, name: str):
        mesh = self.model.mesh(id)
        vertadr = mesh.vertadr.item()
        vertnum = mesh.vertnum.item()
        faceadr = mesh.faceadr.item()
        facenum = mesh.facenum.item()
        vertices = self.model.mesh_vert[vertadr:vertadr + vertnum]
        faces = self.model.mesh_face[faceadr:faceadr + facenum]
        blender_mesh = add_mesh(name, vertices, faces)
        return blender_mesh, np.array((0, 0, 0)), np.array((1, 0, 0, 0)), np.array((1, 1, 1))

def match_and_remove_prefix(text: str, prefixes: list[str]):
    for prefix in prefixes:
        if text.startswith(prefix):
            return text[len(prefix):]
    return None

def add_hierarchy(
    model: mujoco.MjModel,
    mesh_dir: Path,
    gallery: bpy.types.Collection | None = None,
    prefixes: tuple[str] = (),
):
    workspace = bpy.data.collections.new("Workspace")
    bpy.context.scene.collection.children.link(workspace)

    if gallery is None:
        gallery_objects = {}
    else:
        gallery_objects = gallery.objects

    blender_bodies = {}
    body_copied = {}
    for i in range(1, model.nbody):  # skip worldbody
        body = model.body(i)
        identifier = match_and_remove_prefix(body.name, prefixes)
        if identifier is None:
            if "/" in body.name:
                identifier = parse_identifier(body.name)
            else:
                identifier = None
                if model.body_geomnum[i] > 0:
                    print(f"Warning: Unnamespaced body {body.name} with geometry")
        parentid = body.parentid.item()
        if parentid == 0:
            blender_parent = None
        else:
            blender_parent = blender_bodies[parentid]

        if identifier is not None and identifier in gallery_objects:
            blender_body = copy_object_with_geom(
                gallery_objects[identifier],
                body.name[:-len(identifier)],
                parent=blender_parent,
                collection=workspace,
                location=body.pos,
                rotation_quaternion=body.quat,
            )
            body_copied[i] = True
        else:
            blender_body = add_object(
                name=body.name,
                parent=blender_parent,
                collection=workspace,
                location=body.pos,
                rotation_quaternion=body.quat,
            )
            blender_body.empty_display_size = 0.1
            body_copied[i] = False
        blender_bodies[i] = blender_body

    blender_bodies_not_copied = { i: body for i, body in blender_bodies.items() if not body_copied[i] }
    if len(blender_bodies_not_copied) > 0:
        add_geometry(model, blender_bodies_not_copied, workspace, mesh_dir)
    
    blender_cameras = []
    for i in range(model.ncam):
        camera = model.cam(i)
        if camera.mode != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
            continue
        # Assume width/height >= 1
        fovy = camera.fovy.item()
        fovy = np.deg2rad(fovy)
        clip_start = model.vis.map.znear * model.stat.extent
        clip_end = model.vis.map.zfar * model.stat.extent
        camera_data = add_fixed_camera(
            name=camera.name,
            fovy=fovy,
            clip_start=clip_start,
            clip_end=clip_end,
        )
        bodyid = camera.bodyid.item()
        camera_body = blender_bodies.get(bodyid)
        blender_camera = add_object(
            name=camera.name,
            data=camera_data,
            collection=workspace,
            parent=camera_body,
            location=camera.pos,
            rotation_quaternion=camera.quat,
        )
        blender_cameras.append(blender_camera)

    return blender_bodies

def compute_relative_transform(
    pos: np.ndarray, quat: np.ndarray,
    parent_pos: np.ndarray, parent_quat: np.ndarray,
):
    assert pos.ndim == quat.ndim
    assert pos.shape[-1] == 3 and pos.shape == parent_pos.shape
    assert quat.shape[-1] == 4 and quat.shape == parent_quat.shape
    rotation = Rotation.from_quat(quat, scalar_first=True)
    parent_rotation = Rotation.from_quat(parent_quat, scalar_first=True)

    relative_pos = parent_rotation.inv().apply(pos - parent_pos)
    relative_rotation = parent_rotation.inv() * rotation
    relative_quat = relative_rotation.as_quat(scalar_first=True)

    return relative_pos, relative_quat

def compute_parent_transform(
    pos: np.ndarray, quat: np.ndarray,
    relative_pos: np.ndarray, relative_quat: np.ndarray,
):
    assert pos.ndim == quat.ndim
    assert pos.shape[-1] == 3 and pos.shape == relative_pos.shape
    assert quat.shape[-1] == 4 and quat.shape == relative_quat.shape
    rotation = Rotation.from_quat(quat, scalar_first=True)
    relative_rotation = Rotation.from_quat(relative_quat, scalar_first=True)

    parent_rotation = rotation * relative_rotation.inv()
    parent_pos = pos - parent_rotation.apply(relative_pos)
    parent_quat = parent_rotation.as_quat(scalar_first=True)

    return parent_pos, parent_quat

def add_animation(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    blender_bodies: dict[int, bpy.types.Object],
    fps: int,
):
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
    frames = np.arange(num_frames, dtype=np.float64)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = num_frames - 1
    bpy.context.scene.render.fps = fps
    bpy.context.scene.frame_set(0)

    data = mujoco.MjData(model)
    xpos = np.empty((num_frames, model.nbody, 3))
    xquat = np.empty((num_frames, model.nbody, 4))
    for i in range(num_frames):
        data.qpos[:] = qpos[i]
        mujoco.mj_kinematics(model, data)
        xpos[i] = data.xpos
        xquat[i] = data.xquat

    action = bpy.data.actions.new(name="kinematics")
    action_layer = action.layers.new(name="kinematics")
    action_strip = action_layer.strips.new(type="KEYFRAME")
    for i in blender_bodies:
        body = model.body(i)
        if body.weldid != i:
            continue  # welded to animated parent
        parentid = body.parentid.item()
        relative_pos, relative_quat = compute_relative_transform(
            xpos[:, i], xquat[:, i],
            xpos[:, parentid], xquat[:, parentid],
        )

        action_slot = action.slots.new("OBJECT", body.name)
        channelbag = action_strip.channelbags.new(action_slot)
        fcurves = channelbag.fcurves
        jntadr = body.jntadr.item()
        jntnum = body.jntnum.item()
        translation = False
        rotation = False
        for j in range(jntadr, jntadr + jntnum):
            joint = model.joint(j)
            if joint.type == mujoco.mjtJoint.mjJNT_FREE:
                translation = rotation = True
            elif joint.type == mujoco.mjtJoint.mjJNT_BALL:
                rotation = True
                translation = not np.all(joint.pos == 0)
            elif joint.type == mujoco.mjtJoint.mjJNT_SLIDE:
                translation = True
            elif joint.type == mujoco.mjtJoint.mjJNT_HINGE:
                rotation = True
                translation = not np.all(joint.pos == 0)
        if translation:
            for j in range(3):
                fcurve = fcurves.new(data_path="location", index=j)
                fcurve.keyframe_points.add(num_frames)
                co = np.stack((frames, relative_pos[:, j]), axis=-1)
                fcurve.keyframe_points.foreach_set("co", co.ravel())
        else:
            assert np.allclose(relative_pos, body.pos), f"Invalid position for {body.name}"
        if rotation:
            for j in range(4):
                fcurve = fcurves.new(data_path="rotation_quaternion", index=j)
                fcurve.keyframe_points.add(num_frames)
                co = np.stack((frames, relative_quat[:, j]), axis=-1)
                fcurve.keyframe_points.foreach_set("co", co.ravel())
        else:
            assert np.allclose(relative_quat, body.quat), f"Invalid rotation for {body.name}"

        blender_body = blender_bodies[i]
        blender_body.animation_data_create()
        blender_body.animation_data.action = action

def set_keyframe(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    blender_bodies: dict[int, bpy.types.Object],
):
    assert qpos.shape == (model.nq,), f"Invalid qpos shape: {qpos.shape}"
    
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    xpos = data.xpos
    xquat = data.xquat

    for i in blender_bodies:
        body = model.body(i)
        if body.weldid != i:
            continue
        parentid = body.parentid.item()
        relative_pos, relative_quat = compute_relative_transform(
            xpos[i], xquat[i],
            xpos[parentid], xquat[parentid],
        )

        blender_body = blender_bodies[i]
        blender_body.location = relative_pos
        blender_body.rotation_mode = 'QUATERNION'
        blender_body.rotation_quaternion = relative_quat

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

def render_mujoco_from_log(log_dir: Path, gallery_blend: Path | None = None, fps: int = 20):
    log_dir = Path(log_dir)
    model, states, info = load_log(log_dir)
    qpos = take_state_split(states, info['split']['qpos'])

    if gallery_blend is None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        gallery = None
    else:
        bpy.ops.wm.open_mainfile(filepath=str(gallery_blend))
        gallery = bpy.data.collections["Gallery"]

    blender_bodies = add_hierarchy(model, Path("assets"), gallery)
    add_animation(model, qpos, blender_bodies, fps=fps)

    liquid_path = log_dir / "liquid.usd"
    if liquid_path.exists():
        bpy.ops.wm.usd_import(
            filepath=str(liquid_path),
            set_frame_range=False,
            import_cameras=False,
            import_curves=False,
            import_lights=False,
            import_materials=False,
            import_meshes=True,
            import_volumes=False,
            import_shapes=False,
            import_skeletons=False,
            import_blendshapes=False,
            import_points=False,
            import_subdiv=False,
            create_collection=True,
        )

        collection = bpy.context.view_layer.active_layer_collection.collection
        for obj in collection.objects:
            assert obj.type == 'MESH' and obj.name.startswith("liquid_")
            geom_id = int(obj.name[7:])
            geom = model.geom(geom_id)
            body_id = geom.bodyid.item()
            pos = geom.pos
            quat = geom.quat
            assert body_id in blender_bodies
            obj.parent = blender_bodies[body_id]
            obj.location = pos
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = quat
            obj.scale = (0.99, 0.99, 0.99)

            node_group = bpy.data.node_groups['Smooth by Angle']
            modifier = obj.modifiers.new(name="Smooth", type='NODES')
            modifier.node_group = node_group
            material = bpy.data.materials['water']
            obj.data.materials.append(material)
            obj.active_material_index = 1

    if gallery is not None:
        bpy.data.collections.remove(gallery, do_unlink=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(log_dir / "scene.blend"), check_existing=False, compress=True)

    if liquid_path.exists():
        # fix relative path to the blend file
        bpy.data.cache_files["liquid.usd"].filepath = "//liquid.usd"
        bpy.ops.wm.save_mainfile()
    
    if (log_dir / "3.mkv").exists():
        # Hardcoded for now
        ui_material = bpy.data.materials["thermal_mixer_display"]
        texture_node = ui_material.node_tree.nodes["Image Texture"]
        texture_node.image = bpy.data.images.load("//3.mkv")
        texture_node.image_user.frame_start = 0
        texture_node.image_user.frame_duration = bpy.context.scene.frame_end + 1
        texture_node.image_user.frame_offset = 0
        bpy.ops.wm.save_mainfile()

def render_mujoco_from_scene(scene_path: Path, gallery_blend: Path | None = None, keyframe: int = -1):
    scene_path = Path(scene_path)
    model = mujoco.MjModel.from_xml_path(str(scene_path))

    if gallery_blend is None:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        gallery = None
    else:
        bpy.ops.wm.open_mainfile(filepath=str(gallery_blend))
        gallery = bpy.data.collections["Gallery"]

    blender_bodies = add_hierarchy(model, Path("assets"), gallery)
    if keyframe >= 0:
        qpos = model.key_qpos[keyframe]
        set_keyframe(model, qpos, blender_bodies)

    if gallery is not None:
        bpy.data.collections.remove(gallery, do_unlink=True)

    bpy.ops.wm.save_as_mainfile(filepath=str(scene_path.with_suffix(".blend")), check_existing=False, compress=True)        

def parse_identifier(body_name: str):
    # "/" before ":" (if any), separates prefix from identifier
    if ":" in body_name:
        # Namespaced
        prefix, identifier = body_name.split(":", 1)
        if "/" not in prefix:
            print(f"Warning: Missing slash in prefix: {body_name}")
            return None
        else:
            prefix, namespace = prefix.rsplit("/", 1)
            identifier = f"{namespace}:{identifier}"
            parts = identifier.split(":")
            return ":".join(parts[-2:])
    else:
        # Unnamespaced
        if "/" not in body_name:
            print(f"Warning: Missing slash in body name: {body_name}")
            return None
        else:
            prefix, identifier = body_name.rsplit("/", 1)
            return identifier

def remove_prefix(text: str, prefix: str):
    if not text.startswith(prefix):
        raise ValueError(f"Invalid prefix: {text} does not start with {prefix}")
    return text[len(prefix):]

def add_geometry(
    model: mujoco.MjModel,
    blender_bodies: dict[int, bpy.types.Object],
    target_collection: bpy.types.Collection,
    mesh_dir: Path,
    body_prefix: dict[int, str] = None
):
    scratch = ensure_collection("Scratch")
    missing_material = ensure_material("Missing", (1, 0, 1, 1))

    if body_prefix is None:
        body_prefix = { i: "" for i in blender_bodies }

    mesh_manager = MeshManager(model, mesh_dir, scratch)
    materials = {}
    blender_geoms = []
    for i in range(model.ngeom):
        geom = model.geom(i)
        bodyid = geom.bodyid.item()
        if bodyid not in blender_bodies:
            continue
        blender_body = blender_bodies[bodyid]

        pos = geom.pos
        quat = geom.quat

        if geom.group >= 3:
            continue  # skip collision geoms
        if geom.type == mujoco.mjtGeom.mjGEOM_PLANE:
            half_width, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_plane(half_width, half_height)
        elif geom.type == mujoco.mjtGeom.mjGEOM_HFIELD:
            raise NotImplementedError
        elif geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
            radius, _, _ = geom.size
            mesh, scale = mesh_manager.add_sphere(radius)
        elif geom.type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            radius, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_capsule(radius, half_height)
        elif geom.type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            radii = geom.size
            mesh, scale = mesh_manager.add_ellipsoid(radii)
        elif geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            radius, half_height, _ = geom.size
            mesh, scale = mesh_manager.add_cylinder(radius, half_height)
        elif geom.type == mujoco.mjtGeom.mjGEOM_BOX:
            half_extents = geom.size
            mesh, scale = mesh_manager.add_box(half_extents)
        elif geom.type == mujoco.mjtGeom.mjGEOM_MESH or geom.type == mujoco.mjtGeom.mjGEOM_SDF:
            dataid = geom.dataid.item()
            mesh, mesh_pos, mesh_quat, scale = mesh_manager.add_mesh(dataid, body_prefix[bodyid])
            pos, quat = compute_parent_transform(
                pos, quat,
                mesh_pos, mesh_quat,
            )
        else:
            raise NotImplementedError
        matid = geom.matid.item()
        if matid == -1:
            blender_material = missing_material
        elif matid in materials:
            blender_material = materials[matid]
        else:
            material = model.mat(matid)
            material_name = remove_prefix(material.name, body_prefix[bodyid])
            blender_material = add_material(material_name, material.rgba)
            materials[matid] = blender_material

        if geom.name == "":
            name = mesh.name
        else:
            name = remove_prefix(geom.name, body_prefix[bodyid])
        blender_geom = add_object(
            name=name,
            data=mesh,
            parent=blender_body,
            collection=target_collection,
            location=pos,
            rotation_quaternion=quat,
            scale=scale,
            material=blender_material,
            allow_rename=True,
        )
        blender_geoms.append(blender_geom)

    return blender_geoms

def upsert_gallery(model: mujoco.MjModel, mesh_dir: Path, intercept_glb: dict[str, Path]):
    gallery = ensure_collection("Gallery")
    
    blender_bodies = {}
    body_prefix = {}
    seen = set()

    for i in range(1, model.nbody):  # skip worldbody
        body = model.body(i)

        identifier = parse_identifier(body.name)
        prefix = body.name[:-len(identifier)]
        if identifier is None or identifier in seen:
            continue
        seen.add(identifier)
        if identifier in bpy.data.objects:
            print(f"NOTE: Object {identifier} already exists")
            continue
        body_prefix[i] = prefix

        parentid = body.parentid.item()
        if parentid == 0:
            blender_parent = None
        else:
            blender_parent = blender_bodies[parentid]
        blender_body = add_object(
            name=identifier,
            parent=blender_parent,
            collection=gallery,
            location=body.pos,
            rotation_quaternion=body.quat,
        )
        blender_body.empty_display_size = 0.1
        blender_bodies[i] = blender_body
    
    numbered_pattern = re.compile(r"^(.*)-\d+$")
    for name, path in intercept_glb.items():
        matches = [i for i in blender_bodies if blender_bodies[i].name.startswith(name + ":")]
        if len(matches) == 0:
            continue

        import_gltf_mesh(Path(path))
        gltf_objects = bpy.context.selected_objects

        for match in matches:
            blender_body = blender_bodies.pop(match)
            part_name = blender_body.name[len(name) + 1:]
            if numbered_pattern.match(part_name):
                part_name = numbered_pattern.sub(r"\1", part_name)

            pattern = re.compile(rf"^{part_name}-\d+(:?.\d+)?$")
            for obj in gltf_objects:
                if obj.type == 'MESH' and pattern.match(obj.name):
                    break
            else:
                continue
            _, _, scale = obj.matrix_world.decompose()
            blender_geom = add_object(
                name=f"{blender_body.name}-visual",
                data=obj.data,
                parent=blender_body,
                collection=gallery,
                scale=scale,
            )

        for obj in gltf_objects:
            bpy.data.objects.remove(obj, do_unlink=True)

    add_geometry(model, blender_bodies, gallery, mesh_dir, body_prefix)

def build_gallery(gallery_blend: Path, intercept_glb: dict[str, Path]):
    if gallery_blend.exists():
        bpy.ops.wm.open_mainfile(filepath=str(gallery_blend))
        saveas = False
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        saveas = True
    model = mujoco.MjModel.from_xml_path("model/scene/gallery.xml")
    upsert_gallery(model, Path("assets"), intercept_glb)
    if saveas:
        bpy.ops.wm.save_as_mainfile(filepath=str(gallery_blend), check_existing=False, compress=True)
    else:
        bpy.ops.wm.save_mainfile()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--build-gallery", action="store_true")
    parser.add_argument("--gallery-blend", type=Path, default="gallery.blend")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--keyframe", type=int, default=-1)
    parser.add_argument("log_dir", type=Path, nargs="*")

    import sys
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []
    args = parser.parse_args(argv)

    if args.build_gallery:
        intercept_glb = {
            "centrifuge_eppendorf_5430": "../assetlab/workbench/baked/centrifuge_eppendorf_5430.glb",
            "centrifuge_eppendorf_5910_ri": "../assetlab/workbench/baked/centrifuge_eppendorf_5910_ri.glb",
            "centrifuge_tiangen_tgear_mini": "../assetlab/workbench/baked/centrifuge_tiangen_tgear_mini.glb",
            "thermal_cycler_biorad_c1000": "../assetlab/workbench/baked/thermal_cycler_biorad_c1000.glb",
            "thermal_mixer_eppendorf_c": "../assetlab/workbench/baked/thermal_mixer_eppendorf_c.glb",
            "vortex_mixer_genie_2": "../assetlab/workbench/baked/vortex_mixer_genie_2.glb",
        }
        build_gallery(args.gallery_blend, intercept_glb)
    elif args.log_dir is not None:
        if args.static:
            for scene_path in args.log_dir:
                assert scene_path.is_file() and scene_path.suffix == ".xml"
                render_mujoco_from_scene(scene_path, args.gallery_blend, args.keyframe)
        else:
            for log_dir in args.log_dir:
                assert log_dir.is_dir()
                render_mujoco_from_log(log_dir, args.gallery_blend, args.fps)
    else:
        print("Nothing to do")

# blender --background --python render_blender.py -- --build-gallery
# blender --background --python render_blender.py -- logs/pickup/2025-04-09_13-48-47
