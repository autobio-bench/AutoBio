# TODO: support insertion

from dataclasses import dataclass

import numpy as np
import trimesh
from trimesh import Trimesh
import mujoco

from meshplane import Mesh, MeshPlane, SurfaceDynamics

from simulation import System

Z = np.array([0, 0, 1], dtype=np.float64)

@dataclass
class ContainerDefinition:
    interior: Mesh
    exterior: Mesh | None

    _interior: Trimesh
    _exterior: Trimesh | None
    _interior_opening_vertex_mask: np.ndarray
    _exterior_opening_vertex_mask: np.ndarray | None
    opening_direction: np.ndarray

    @staticmethod
    def from_object_mesh(
        mesh: Trimesh,
        *,
        split_top: bool = True,
        split_bottom: bool = False,
        opening: str = "top",
        threshold: float = 0.01,
    ) -> "ContainerDefinition":
        # Create a container definition by splitting the mesh into two parts
        # and optionally capping the top and bottom

        assert mesh.is_volume, "Mesh must be a volume mesh"

        mesh = mesh.copy()
        bottom, top = mesh.bounds[:, 2]
        bottom_threshold = bottom + (top - bottom) * threshold
        top_threshold = top - (top - bottom) * threshold

        if split_top:
            top_vertex_mask = mesh.vertices[:, 2] > top_threshold
            top_face_mask = np.all(top_vertex_mask[mesh.faces], axis=-1)
            mesh.update_faces(~top_face_mask)

        if split_bottom:
            bottom_vertex_mask = mesh.vertices[:, 2] < bottom_threshold
            bottom_face_mask = np.all(bottom_vertex_mask[mesh.faces], axis=-1)
            mesh.update_faces(~bottom_face_mask)
        
        parts = mesh.split(only_watertight=False)
        assert len(parts) == 2, "Mesh must be split into two parts"
        parts.sort(key=lambda part: part.area)
        interior, exterior = parts
        interior.invert()

        interior = interior.convex_hull
        exterior = exterior.convex_hull

        if opening == "top":
            interior_opening_vertex_mask = interior.vertices[:, 2] > top_threshold
            exterior_opening_vertex_mask = exterior.vertices[:, 2] > top_threshold
            opening_direction = np.array([0, 0, 1], dtype=np.float64)
        elif opening == "bottom":
            interior_opening_vertex_mask = interior.vertices[:, 2] < bottom_threshold
            exterior_opening_vertex_mask = exterior.vertices[:, 2] < bottom_threshold
            opening_direction = np.array([0, 0, -1], dtype=np.float64)
        else:
            raise ValueError("Invalid opening type")
        
        interior_mesh = Mesh(interior.vertices, interior.faces.astype(np.uint64), np.where(interior_opening_vertex_mask)[0].astype(np.uint64))
        # exterior_mesh = Mesh(exterior.vertices, exterior.faces.astype(np.uint64), np.where(exterior_opening_vertex_mask)[0].astype(np.uint64))

        return ContainerDefinition(
            interior=interior_mesh,
            exterior=None,
            _interior=interior,
            _exterior=exterior,
            _interior_opening_vertex_mask=interior_opening_vertex_mask,
            _exterior_opening_vertex_mask=exterior_opening_vertex_mask,
            opening_direction=opening_direction,
        )

@dataclass
class LiquidSurface:
    valid: bool
    distance: float
    center: np.ndarray
    frame: np.ndarray
    half_width: float
    half_height: float

@dataclass
class LiquidState:
    meshplane: MeshPlane
    max_volume: float

    volume: float

    surface_dynamics: SurfaceDynamics
    surface: LiquidSurface

    @property
    def surface_normal(self) -> np.ndarray:
        return self.surface_dynamics.normal
    
    @staticmethod
    def create(mesh: Mesh, volume: float, acceleration: np.ndarray, dt: float, *, lambda_phi: float = 0.03, lambda_theta: float = 0.03):
        meshplane = MeshPlane(mesh)
        surface_normal = acceleration / np.linalg.norm(acceleration)
        meshplane.set_plane_normal(*surface_normal)
        low, high, _ = meshplane.get_plane_distance_range()
        ls = LiquidState(
            meshplane=meshplane,
            max_volume=mesh.volume,
            volume=volume,
            surface_dynamics=SurfaceDynamics(surface_normal, dt, lambda_phi, lambda_theta),
            surface= LiquidSurface(
                valid=False,
                distance=(low + high) / 2,  # Initial guess for distance
                center=None,
                frame=None,
                half_width=None,
                half_height=None,
            )
        )
        # ls.surface_normal = surface_normal
        return ls

    def update_normal(self, volume: float, acceleration: np.ndarray):
        # In steady state, the liquid surface normal is equal to the acceleration direction
        if volume > self.max_volume * 0.9 or volume < 0:
            raise ValueError(f"Volume is out of range, expected [0, {self.max_volume} * 0.9], got {volume}")
        self.volume = volume

        # from scipy.integrate import solve_ivp
        # self.surface_dynamics.gravity = acceleration
        # self.surface_dynamics.length = (self.surface.half_width * self.surface.half_height) ** 0.5 * 3
        # sol = solve_ivp(
        #     lambda _, y: self.surface_dynamics.dynamics(y),
        #     (0, 0.002),
        #     self.surface_dynamics.state,
        # )
        # self.surface_dynamics.state = sol.y[:, -1]

        length = (self.surface.half_width * self.surface.half_height) ** 0.5
        self.surface_dynamics.step(acceleration, length * 3)
        self.meshplane.set_plane_normal(*self.surface_normal)

    def update_level(self):
        low, high, soft_high = self.meshplane.get_plane_distance_range()
        distance = self.meshplane.solve_plane_distance(self.volume, self.surface.distance)

        valid = distance < soft_high
        result = self.meshplane.calculate_plane(distance)

        self.surface = LiquidSurface(
            valid=valid,
            distance=distance,
            center=result.center,
            frame=result.frame,
            half_width=result.half_width,
            half_height=result.half_height,
        )


@dataclass
class Container:
    definition: ContainerDefinition

    position: np.ndarray
    rotation_matrix: np.ndarray

    liquid: LiquidState | None

    will_insert: bool
    insert_target: "Container | None"

    dt: float

    @property
    def volume(self) -> float:
        if self.liquid is None:
            return 0
        return self.liquid.volume

    def update(self, position: np.ndarray, rotation_matrix: np.ndarray, acceleration: np.ndarray, volume: float = None):
        self.position = position
        self.rotation_matrix = rotation_matrix
        if volume is None:
            volume = self.volume
        if volume == 0:
            self.liquid = None
            return

        local_acceleration = self.rotation_matrix.T @ acceleration
        if self.liquid is None:
            self.liquid = LiquidState.create(self.definition.interior, volume=volume, acceleration=local_acceleration, dt=self.dt)
        else:
            self.liquid.update_normal(volume, local_acceleration)

        self.liquid.update_level()

# @dataclass
# class Channel:
#     ...

@dataclass
class LiquidLog:
    geom_id: int
    vertices: np.ndarray
    faces: np.ndarray
    boundary: np.ndarray
    normal: list[np.ndarray]
    distance: list[float]


class ContainerSystem(System):
    def _configure(
        self, *,
        definition: ContainerDefinition | None = None,
        split_top: bool = True,
        split_bottom: bool = False,
        opening: str = "top",
        initial_volume: float = None,
    ):
        self.custom_definition = definition
        self.split_top = split_top
        self.split_bottom = split_bottom
        self.opening = opening
        self.initial_volume = initial_volume
    
    def _reload(self, model: mujoco.MjModel):
        self.geom_id = self.name2id(mujoco.mjtObj.mjOBJ_GEOM)
        if self.custom_definition is None:
            # Deduce the mesh from the geom
            assert model.geom_type[self.geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            data_id = model.geom_dataid[self.geom_id].item()
            assert data_id != -1
            vertadr = model.mesh_vertadr[data_id].item()
            vertnum = model.mesh_vertnum[data_id].item()
            faceadr = model.mesh_faceadr[data_id].item()
            facenum = model.mesh_facenum[data_id].item()
            assert vertadr != -1 and vertnum > 0 and faceadr != -1 and facenum > 0
            vertices = model.mesh_vert[vertadr:vertadr + vertnum]
            faces = model.mesh_face[faceadr:faceadr + facenum]
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
            self.definition = ContainerDefinition.from_object_mesh(
                mesh,
                split_top=self.split_top,
                split_bottom=self.split_bottom,
                opening=self.opening,
            )
        else:
            self.definition = self.custom_definition
        self.gravity = model.opt.gravity
        self.log = LiquidLog(
            geom_id=self.geom_id,
            vertices=self.definition._interior.vertices,
            faces=self.definition._interior.faces,
            boundary=np.where(self.definition._interior_opening_vertex_mask)[0].astype(np.uint64),
            normal=[],
            distance=[],
        )

    def _reset(self, data: mujoco.MjData):
        self.container = Container(
            definition=self.definition,
            position=None,
            rotation_matrix=None,
            liquid=None,
            will_insert=False,
            insert_target=None,
            dt=self.model.opt.timestep,
        )
        self.log.normal = []
        self.log.distance = []
        if self.initial_volume is None:
            volume = self.container.definition.interior.volume * 0.5
        else:
            volume = self.initial_volume
        self._update_container(data, volume=volume)

    def _update(self, data: mujoco.MjData):
        self._update_container(data)

    def _update_container(self, data: mujoco.MjData, volume: float = None):
        acceleration = np.zeros(6)  # rot:lin
        mujoco.mj_objectAcceleration(
            self.model, data,
            mujoco.mjtObj.mjOBJ_GEOM,
            self.geom_id,
            acceleration,
            False,
        )
        acceleration = acceleration[3:]  # only linear acceleration
        self.container.update(
            data.geom_xpos[self.geom_id],
            data.geom_xmat[self.geom_id].reshape(3, 3),
            acceleration,
            volume=volume,
        )
        self.log.normal.append(self.container.liquid.surface_normal)
        self.log.distance.append(self.container.liquid.surface.distance)

    def _visualize(self, data: mujoco.MjData, scene: mujoco.MjvScene):
        # Visualize water surface
        if self.container.liquid is None:
            return
        
        surface = self.container.liquid.surface
        position = self.container.position
        rotation_matrix = self.container.rotation_matrix
        local_center = surface.center
        local_frame = surface.frame
        half_width = surface.half_width
        half_height = surface.half_height
        world_center = rotation_matrix @ local_center + position
        world_frame = rotation_matrix @ local_frame
        if not surface.valid:
            rgba = (1, 0, 0, 1)
        else:
            rgba = (0, 0, 1, 1)

        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(half_width, half_height, 1),
            pos=world_center,
            mat=world_frame.ravel(),
            rgba=rgba,
        )
        scene.ngeom += 1

class ContainerCoordinator(System):
    def _configure(self):
        self._need_manager = True

    def _reload(self, model: mujoco.MjModel):
        self._css: list[ContainerSystem] = self.manager.systems_by_type.get(ContainerSystem, [])

    def _finish(self):
        liquids = []
        for cs in self._css:
            log = {
                "geom_id": cs.geom_id,
                "vertices": cs.log.vertices,
                "faces": cs.log.faces,
                "boundary": cs.log.boundary,
                "normal": np.stack(cs.log.normal),
                "distance": np.stack(cs.log.distance),
            }
            liquids.append(log)
        with open("liquid.pkl", "wb") as f:
            import pickle
            pickle.dump(liquids, f)
