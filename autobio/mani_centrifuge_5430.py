import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

import numpy as np
from kinematics import IK, Pose, slerp, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import Centrifuge_Eppendorf_5430


def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

class UR5eArm:

    # 6-DOF arm

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}shoulder_pan'
        self.act_name = f'{prefix}shoulder_pan'
        self.site_name = f'{prefix}2f85:pinch'
        self.base_name = f'{prefix}base'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f'{prefix}2f85:fingers_actuator').id
        self.gripper_jnt_adr = model.joint(f'{prefix}2f85:right_driver_joint').qposadr.item()
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)
    
    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self):
        lows = (-0.1, 0.0, -0.2, -0.1, 0.0, -0.2)
        highs = (0.1, 0.3, 0.2, 0.1, 0.3,  0.2)
        perturbation = np.random.uniform(lows, highs)
        return perturbation


class Centrifuge_5430(Centrifuge_Eppendorf_5430):

    def _reset(self, data):
        super()._reset(data)
        self.fk_lever = FK(1, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid')

    def fk(self, qpos: np.ndarray) -> Pose:
        return self.fk_lever.forward(qpos)
    
    def qpos_interpolate(self, qpos_list: list[np.ndarray], num_steps: list[int]) -> list[np.ndarray]:
        interpolated_qpos = []
        for i in range(len(num_steps)):
            start_qpos = qpos_list[i]
            end_qpos = qpos_list[i + 1]
            step_qpos = (end_qpos - start_qpos) / num_steps[i]
            for step in range(num_steps[i]):
                interpolated_qpos.append(start_qpos + step * step_qpos)
        interpolated_qpos.append(qpos_list[-1])
        return interpolated_qpos

    def lever_path(self, data: mujoco.MjData, mode: str='1/close') -> list[Pose]:
        # site path for open/close the lid
        cur_qpos = data.qpos[self.lid_qposadr]
        cur_qpos_arr = np.asarray(cur_qpos).reshape(1)
        match mode:
            case '1/close':
                qpos1 = np.array([self.lid_jntlimit[0] + 0.1592037]) 
                qpos_list = [cur_qpos_arr, qpos1]
                num_steps = [15]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), 'grip') for qpos in qpos_list]
                path.append(self.get_eefpose_lever(self.fk(qpos1), '2/detach'))
            case _:
                raise ValueError(f"Unknown lever path mode: {mode}")
        return path

    def get_eefpose_lever(self, sitepose: Pose, mode: str='1/detach') -> Pose:
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi)
        match mode:
            case '1/detach':
                rel_pos = np.array([0.0, 0.008, 0.017])
            case '2/detach':
                rel_pos = np.array([0.0, 0.008, 0.03])
            case 'grip':
                # rel_pos = np.zeros(3)
                rel_pos = np.array([0.0, 0.0, -0.005])
            case 'lock_pre':
                # rel_pos = np.zeros(3)
                rel_pos = np.array([0.0, 0.2, -0.017])
            case 'lock':
                # rel_pos = np.zeros(3)
                rel_pos = np.array([0.0, -0.045, -0.06])
            case _:
                print("failure", mode)
                raise ValueError(f"Unknown approach mode: {mode}")
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)

    def get_eef_pose(self, data: mujoco.MjData, loc: str, mode: str='1/detach', random: bool=False) -> Pose:
        match loc:
            case 'lid':
                site_pos = data.site_xpos[self.lid_site]
                site_mat = data.site_xmat[self.lid_site]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, site_mat)
                return self.get_eefpose_lever(Pose(site_pos, quat), mode)
            case _:
                raise ValueError(f"Unknown location: {loc}")


class Centrifuge5430Manipulate(Task):
    default_scene = SCENE_ROOT / "mani_centrifuge_5430.xml"
    default_task = "centrifuge_5430_close_lid"

    time_limit = 15.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = Centrifuge_5430('/centrifuge_eppendorf_5430:')
        manager = Manager.from_spec(spec, [self.instrument])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        self.task_info = {
            'prefix': 'close the lid of the centrifuge eppendorf 5430',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {
                'image': 'table_cam_left',
                'wrist_image': '/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        # Add any task-specific completion checks if needed
        return True

class Centrifuge5430ManipulateExpert(Centrifuge5430Manipulate, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=1.5,
            qc_acc=1.0,
            ik=self.arm.ik.solve
        )

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def interpolate2(self, start: Pose, end: Pose, num_steps: int, height: float = None) -> list[Pose]:
        path = []
        p1 = start.pos
        p2 = end.pos
        horizon_vec = np.array([p2[0]-p1[0], p2[1]-p1[1], 0.0])
        horizon_dis = np.linalg.norm(horizon_vec)
        origin = p1.copy()
        origin[2] = 0.0
        basis1 = horizon_vec / horizon_dis
        basis2 = np.array([0.0, 0.0, 1.0])
        p1_ = np.array([0.0, p1[2]])
        p2_ = np.array([horizon_dis, p2[2]])
        if height is None:
            height = horizon_dis / 4.0
        p3_ = (p1_ + p2_) / 2.0
        p3_[1] += height
        x = np.array([p1_[0], p3_[0], p2_[0]])
        y = np.array([p1_[1], p3_[1], p2_[1]])
        coef = np.polyfit(x, y, 2)
        x_eval = np.linspace(p1_[0], p2_[0], num_steps + 1)
        y_eval = np.polyval(coef, x_eval)
        for i in range(num_steps + 1):
            t = i / num_steps
            quat = slerp(start.quat, end.quat, t)
            pos = x_eval[i] * basis1 + y_eval[i] * basis2 + origin
            path.append(Pose(pos, quat))
        return path

    def path_follow(self, path: list[Pose]):
        trajectory = self.planner.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = self.planner.query(trajectory, t)
                self.data.ctrl[self.arm.act_span] = ctrl
            self.step_and_log({})
    
    def move_to(self, pose: Pose, num_steps: int = 2):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def gripper_control(self, value: float):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(300):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        if self.task == 'centrifuge_5430_close_lid':
            target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='grip')
            self.move_to(target_pose, num_steps=2)
            self.gripper_control(240)

            path = self.instrument.lever_path(self.data, mode='1/close')
            self.path_follow(path[:-1])
            self.gripper_control(100)
            self.move_to(path[-1], num_steps=5)

            lock_quat = np.array([0.0, 0.7071, 0.7071, 0.0])
            lock_pose_pre = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock_pre')
            self.gripper_control(255)
            lock_pose_pre.quat = lock_quat
            self.move_to(lock_pose_pre, num_steps=5)

            lock_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock')
            lock_pose.quat = lock_quat
            self.move_to(lock_pose, num_steps=5)

            final_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock_pre')
            final_pose.quat = lock_quat
            self.move_to(final_pose, num_steps=5)

        self.finish()

Centrifuge5430Manipulate.Expert = Centrifuge5430ManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = Centrifuge5430Manipulate.load()
    expert = Centrifuge5430Manipulate.Expert(spec)
    for i in trange(1):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
