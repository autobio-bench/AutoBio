from dataclasses import dataclass

import numpy as np
import mujoco

from grasp.geom import Geom, build_geom
from grasp.site import Site, build_site
from grasp.joint import FreeJoint, Joint, build_joint
from grasp.transform import Transform


@dataclass
class Body:
    id: int
    parent: int
    name: str
    mass: float
    inertia: np.ndarray
    initial_transform: Transform
    joints: list[Joint]
    geoms: list[Geom]
    sites: list[Site]

    @property
    def nq(self):
        return sum(j.nq for j in self.joints)

    @property
    def joint(self):
        if len(self.joints) == 1:
            return self.joints[0]
        raise ValueError(f'Body {self.name} has {len(self.joints)}>1 joints')

    @property
    def visual_geoms(self):
        return [geom for geom in self.geoms if geom.visual]

    @property
    def collision_geoms(self):
        return [geom for geom in self.geoms if geom.collision]

    @property
    def geom(self):
        if len(self.collision_geoms) == 1:
            return self.collision_geoms[0]
        raise ValueError(f'Body {self.name} has {len(self.geoms)}>1 geoms')

    @property
    def is_free(self):
        return len(self.joints) == 1 and isinstance(self.joints[0], FreeJoint)

def build_body(model: mujoco.MjModel, i: int, root: bool, visual_groups: tuple[int, ...], collision_groups: tuple[int, ...]) -> Body:
    parent = model.body_parentid[i].item()
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = model.body_mass[i].item()
    inertia = model.body_inertia[i]
    initial_transform = Transform(model.body_pos[i], model.body_quat[i])
    joint_num = model.body_jntnum[i].item()
    joint_adr = model.body_jntadr[i].item()
    if root and joint_num == 1:
        joints = [build_joint(model, joint_adr, allow_free=True)]
        if isinstance(joints[0], FreeJoint):
            initial_transform = Transform(np.zeros(3), np.array([1, 0, 0, 0]))
    else:
        joints = [build_joint(model, joint_adr + j) for j in range(joint_num)]
    geom_num = model.body_geomnum[i].item()
    geom_adr = model.body_geomadr[i].item()
    geoms = [build_geom(model, geom_adr + j, visual_groups, collision_groups) for j in range(geom_num)]
    geoms = [geom for geom in geoms if geom is not None]
    sites = [build_site(model, j) for j in range(model.nsite) if model.site_bodyid[j] == i]
    return Body(i, parent, name, mass, inertia, initial_transform, joints, geoms, sites)
