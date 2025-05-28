from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from grasp.quat import quat2rot, quatcompose, quatapply, quatinv


@dataclass
class Transform:
    """Rigid body transform. When used in world frame, it can be seen as a pose."""
    pos: np.ndarray
    quat: np.ndarray

    @property
    def rotmat(self):
        return quat2rot(self.quat)

    @property
    def mat(self):
        mat = np.zeros((4, 4))
        mat[:3, :3] = self.rotmat
        mat[:3, 3] = self.pos
        mat[3, 3] = 1
        return mat

    @staticmethod
    def compose(*transforms: 'Transform'):
        """A * B * C * ..., where the last one is applied first"""
        assert len(transforms) > 0
        pos = transforms[-1].pos
        quat = transforms[-1].quat
        for p in reversed(transforms[:-1]):
            pos = quatapply(p.quat, pos) + p.pos
            quat = quatcompose(p.quat, quat)
        return Transform(pos, quat)

    def apply(self, point):
        return quatapply(self.quat, point) + self.pos

    def apply_inv(self, point):
        return quatapply(quatinv(self.quat), point - self.pos)

    def inverse(self):
        return Transform(-quatapply(quatinv(self.quat), self.pos), quatinv(self.quat))

    def square_distance(self, other: 'Transform'):
        def dot(x, y):
            return sum(t * o for t, o in zip(x, y))

        pos_error = self.pos - other.pos
        pos_distance = dot(pos_error, pos_error)
        quat_distance = 1 - dot(self.quat, other.quat)**2
        return pos_distance + quat_distance

    identity: ClassVar['Transform']

Transform.identity = Transform(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
