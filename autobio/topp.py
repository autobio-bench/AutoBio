from typing import Callable, List
from dataclasses import dataclass
import numpy as np
import toppra as ta
from kinematics import Pose

class Topp:
    # re-parameterize trajectory using TOPP-RA

    def __init__(self, dof: int, qc_vel: float, qc_acc: float, ik: Callable):
        self.dof = dof
        self.qc_vel = [(-qc_vel, qc_vel)] * self.dof
        self.qc_vel = ta.constraint.JointVelocityConstraint(np.array(self.qc_vel))
        self.qc_acc = [(-qc_acc, qc_acc)] * self.dof
        self.qc_acc = ta.constraint.JointAccelerationConstraint(np.array(self.qc_acc))
        self.ik: Callable = ik
    
    def jnt_traj(self, pose_path: List[Pose],):
        assert self.ik is not None, "IK solver not set"
        ss = np.linspace(0, 1, len(pose_path))
        jnts = [self.ik(pose.pos, pose.quat) for pose in pose_path]
        path = ta.SplineInterpolator(ss, jnts)
        instance = ta.algorithm.TOPPRA([self.qc_vel, self.qc_acc], path)
        return instance.compute_trajectory(0, 0)

    @staticmethod
    def query(traj: ta.interpolator.AbstractGeometricPath, t: float):
        t = np.clip(t, 0, traj.duration)
        return traj.eval(t)