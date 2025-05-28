from dataclasses import dataclass

import numpy as np
import mujoco

from grasp.transform import Transform

@dataclass
class Site:
    id: int
    name: str
    transform: Transform

def build_site(model: mujoco.MjModel, i: int) -> Site:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
    return Site(i, name, Transform(model.site_pos[i], model.site_quat[i]))
