from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import mujoco

from grasp.transform import Transform
from grasp.quat import axisangle2quat, quatapply

@dataclass
class Joint(ABC):
    nq: ClassVar[int]
    id: int
    name: str

    @abstractmethod
    def to_transform(self, q: np.ndarray) -> Transform:
        ...

    @property
    @abstractmethod
    def bound(self):
        ...

@dataclass
class FreeJoint(Joint):
    nq: ClassVar[int] = 7

    def to_transform(self, q: np.ndarray):
        assert len(q) == FreeJoint.nq
        pos = q[:3]
        quat = q[3:]
        return Transform(pos, quat)

    @property
    def bound(self):
        return [(-np.inf, np.inf)] * 7

@dataclass
class BallJoint(Joint):
    nq: ClassVar[int] = 4
    pos: np.ndarray
    high: float

    def to_transform(self, q: np.ndarray):
        assert len(q) == BallJoint.nq
        quat = q
        pos = self.pos - quatapply(quat, self.pos)
        return Transform(pos, quat)

    @property
    def bound(self):
        raise NotImplementedError

@dataclass
class SlideJoint(Joint):
    nq: ClassVar[int] = 1
    axis: np.ndarray
    low: float
    high: float

    def to_transform(self, q: np.ndarray):
        assert len(q) == SlideJoint.nq
        d = q[0]
        pos = self.axis * d
        quat = np.array([1, 0, 0, 0])
        return Transform(pos, quat)

    @property
    def bound(self):
        return [(self.low, self.high)]

@dataclass
class HingeJoint(Joint):
    nq: ClassVar[int] = 1
    pos: np.ndarray
    axis: np.ndarray
    low: float
    high: float

    def to_transform(self, q: np.ndarray):
        assert len(q) == HingeJoint.nq
        angle = q[0]
        quat = axisangle2quat(self.axis, angle)
        pos = self.pos - quatapply(quat, self.pos)
        return Transform(pos, quat)
    
    @property
    def bound(self):
        return [(self.low, self.high)]

def build_joint(model: mujoco.MjModel, i: int, allow_free: bool = False) -> Joint:
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    joint_type = model.jnt_type[i]
    if joint_type == mujoco.mjtJoint.mjJNT_FREE:
        if not allow_free:
            raise ValueError('Free joint is not allowed here')
        return FreeJoint(i, joint_name)
    elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
        assert model.jnt_limited[i]
        joint_pos = model.jnt_pos[i]
        joint_range = model.jnt_range[i]
        joint_high = model.jnt_range[i][1].item()
        return BallJoint(i, joint_name, joint_pos, joint_high)
    elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
        assert model.jnt_limited[i]
        joint_axis = model.jnt_axis[i]
        joint_range = model.jnt_range[i]
        joint_low, joint_high = joint_range.tolist()
        return SlideJoint(i, joint_name, joint_axis, joint_low, joint_high)
    elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
        joint_pos = model.jnt_pos[i]
        joint_axis = model.jnt_axis[i]
        joint_range = model.jnt_range[i]
        if model.jnt_limited[i]:
            joint_low, joint_high = joint_range.tolist()
        else:
            joint_low, joint_high = -np.pi, np.pi
        return HingeJoint(i, joint_name, joint_pos, joint_axis, joint_low, joint_high)
    else:
        raise ValueError(f'Unsupported joint type: {joint_type}')
