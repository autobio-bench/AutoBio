from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import mujoco

from grasp.transform import Transform

if TYPE_CHECKING:
    from trimesh.base import Trimesh
    from grasp.sdf import SDF

@dataclass
class Geom(ABC):
    id: int
    name: str
    transform: Transform
    rgba: np.ndarray
    visual: bool
    collision: bool

    @abstractmethod
    def as_trimesh(self, transform=None) -> 'Trimesh':
        ...

    def as_sdf(self) -> 'SDF':
        raise NotImplementedError

@dataclass
class Sphere(Geom):
    radius: float

    def as_trimesh(self, transform=None):
        from trimesh.primitives import Sphere
        sphere = Sphere(radius=self.radius, transform=transform)
        sphere.visual.face_colors = self.rgba[:3] * 255
        return sphere

    def as_sdf(self):
        from grasp.sdf import Sphere
        return Sphere(self.radius)

@dataclass
class Capsule(Geom):
    radius: float
    half_height: float

    def as_trimesh(self, transform=None):
        from trimesh.primitives import Capsule
        capsule = Capsule(radius=self.radius, height=2 * self.half_height, transform=transform)
        capsule.visual.face_colors = self.rgba[:3] * 255
        return capsule

    def as_sdf(self):
        from grasp.sdf import Capsule
        return Capsule(self.half_height, self.radius)

@dataclass
class Ellipsoid(Geom):
    radii: np.ndarray

@dataclass
class Cylinder(Geom):
    radius: float
    half_height: float

    def as_trimesh(self, transform=None):
        from trimesh.primitives import Cylinder
        cylinder = Cylinder(radius=self.radius, height=2 * self.half_height, transform=transform)
        cylinder.visual.face_colors = self.rgba[:3] * 255
        return cylinder

@dataclass
class Box(Geom):
    half_sizes: np.ndarray

    def as_trimesh(self, transform=None):
        from trimesh.primitives import Box
        box = Box(self.half_sizes * 2, transform=transform)
        box.visual.face_colors = self.rgba[:3] * 255
        return box

    def as_sdf(self):
        from grasp.sdf import Box
        return Box(*self.half_sizes)

class ConciseArray(np.ndarray):
    def __repr__(self):
        return f'{self.dtype.name}[{",".join(str(x) for x in self.shape)}]'

@dataclass
class Mesh(Geom):
    mesh_name: str
    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        self.vertices = self.vertices.view(ConciseArray)
        self.faces = self.faces.view(ConciseArray)

    def as_trimesh(self, transform=None):
        from trimesh.base import Trimesh
        mesh = Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        mesh.visual.face_colors = self.rgba[:3] * 255
        if transform is not None:
            mesh.apply_transform(transform)
        return mesh

DEFAULT_RGBA = np.array([0.5, 0.5, 0.5, 1.0])
def build_geom(model: mujoco.MjModel, i: int, visual_groups: tuple[int, ...], collision_groups: tuple[int, ...]) -> Geom:
    geom_group = model.geom_group[i].item()
    geom_visual, geom_collision = False, False
    if geom_group in visual_groups:
        geom_visual = True
    elif geom_group in collision_groups:
        geom_collision = True
    if not geom_visual and not geom_collision:
        print(f'Warning: geom {i} has unknown group {geom_group}')
        return None
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    geom_type = model.geom_type[i]
    geom_size = model.geom_size[i]
    geom_rgba = model.geom_rgba[i]
    if np.array_equal(geom_rgba, DEFAULT_RGBA):
        geom_matid = model.geom_matid[i].item()
        if geom_matid >= 0:
            geom_rgba = model.mat_rgba[geom_matid]
    geom_pos = model.geom_pos[i]
    geom_quat = model.geom_quat[i]
    geom_transform = Transform(geom_pos, geom_quat)
    if geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
        raise NotImplementedError
    elif geom_type == mujoco.mjtGeom.mjGEOM_HFIELD:
        raise NotImplementedError
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        radius = geom_size[0].item()
        return Sphere(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, radius)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        radius = geom_size[0].item()
        half_height = geom_size[1].item()
        return Capsule(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, radius, half_height)
    elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        radii = geom_size
        return Ellipsoid(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, radii)
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        radius = geom_size[0].item()
        half_height = geom_size[1].item()
        return Cylinder(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, radius, half_height)
    elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        half_sizes = geom_size
        return Box(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, half_sizes)
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        # TODO: Proper reuse if needed
        geom_dataid = model.geom_dataid[i].item()
        assert geom_dataid >= 0
        mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, geom_dataid)
        mesh_vertadr = model.mesh_vertadr[geom_dataid].item()
        mesh_vertnum = model.mesh_vertnum[geom_dataid].item()
        mesh_faceadr = model.mesh_faceadr[geom_dataid].item()
        mesh_facenum = model.mesh_facenum[geom_dataid].item()
        mesh_vert = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum]
        mesh_face = model.mesh_face[mesh_faceadr:mesh_faceadr + mesh_facenum]
        return Mesh(i, geom_name, geom_transform, geom_rgba, geom_visual, geom_collision, mesh_name, mesh_vert, mesh_face)
    elif geom_type == mujoco.mjtGeom.mjGEOM_SDF:
        raise NotImplementedError
    else:
        raise ValueError(f'Unsupported geom type: {geom_type}')
