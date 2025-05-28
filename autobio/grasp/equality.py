from dataclasses import dataclass

import numpy as np
import mujoco

@dataclass
class JointEquality:
    id: int
    name: str
    active: bool
    joint1: int
    joint2: int | None
    polycoef: np.ndarray

    def compute_joint1(self, joint2):
        return (
            self.polycoef[0] +
            self.polycoef[1] * joint2 +
            self.polycoef[2] * joint2**2 +
            self.polycoef[3] * joint2**3 +
            self.polycoef[4] * joint2**4
        )

def build_equality(model: mujoco.MjModel, i: int) -> JointEquality:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i)
    eq_type = model.eq_type[i]
    if eq_type != mujoco.mjtEq.mjEQ_JOINT:
        # print(f"Skipping non-joint equality constraint {i}")
        return
    active = bool(model.eq_active0[i])
    joint1 = model.eq_obj1id[i].item()
    joint2 = model.eq_obj2id[i].item()
    if joint2 == -1:
        joint2 = None
    else:
        assert joint1 > joint2
    polycoef = model.eq_data[i][:5]
    return JointEquality(i, name, active, joint1, joint2, polycoef)
