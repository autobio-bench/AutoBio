import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
import numpy as np
import math
from kinematics import IK, Pose, slerp, mul_pose, neg_pose
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import Centrifuge_Eppendorf_5430

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

class GridSlot:

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.grids = dict()
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if not site_name.startswith(prefix):
                continue
            parts = site_name.split('-')
            slot_type = parts[1]
            if slot_type not in self.grids:
                info = dict()
                info['origin'] = i
                user = self.model.site_user[i]
                info['rows'] = user[1]
                info['row_gap'] = user[2]
                info['cols'] = user[3]
                info['col_gap'] = user[4]
                info['height'] = user[5]
                self.grids[slot_type] = info
            else:
                print(f"Duplicate grid slot type '{slot_type}' found in '{site_name}'")
        if len(self.grids) == 0:
            raise ValueError(f"No grid slots found with prefix '{prefix}'")

    def get_position(self, data: mujoco.MjData, row: int, col: int, slot_type: str='default', hei: int=0) -> np.ndarray:
        grid = self.grids[slot_type]
        origin = data.site_xpos[grid['origin']]
        frame = data.site_xmat[grid['origin']].reshape(3, 3)
        row_direction = frame[:, 0]
        col_direction = frame[:, 1]
        bias = row * grid['row_gap'] * row_direction + col * grid['col_gap'] * col_direction + np.array([0.0, 0.0, grid['height'] * hei])
        return origin + bias


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
    
    def get_tube_pose(self, data: mujoco.MjData) -> Pose:
        sitepose = self.get_site_pose(data)
        rel_pos = np.array([0.0, 0.0, 0.06])
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, [0.0, 1.0, 0.0], np.pi)
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)

    def qpos_perturb(self):
        lows = (-1.2, -0.2, -0.1, -0.5, -0.2, -0.2)
        highs = (0.0, 0.0, 0.1, 0.2, 0.2,  0.2)
        perturbation = np.random.uniform(lows, highs)
        return perturbation


class CentrifugeTube:
    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str):
        self.model = model
        cap = model.body(f'{cap_prefix}centrifuge_1-5ml_screw_cap')
        body = model.body(f'{body_prefix}centrifuge_1-5ml_screw_body')
        self.cap_id = cap.id
        self.body_id = body.id
        root_id = body.weldid.item()
        root = model.body(root_id)
        self.jnt_adr = model.joint(root.jntadr.item()).qposadr.item()
        self.pos_span = range(self.jnt_adr, self.jnt_adr + 3)
        self.quat_span = range(self.jnt_adr + 3, self.jnt_adr + 7)
    
    def get_pose(self, data: mujoco.MjData) -> Pose:
        pos = data.qpos[self.pos_span]
        quat = data.qpos[self.quat_span]
        return Pose(pos, quat)

    def set_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span] = pose.pos
        data.qpos[self.quat_span] = pose.quat

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def get_eef_pose(self, data: mujoco.MjData) -> Pose:
        cap_pose = self.get_cap_pose(data)
        body_pose = self.get_body_pose(data)
        rel_pos = np.array([0.01, 0.0, 0.01])
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, [0.0, 1.0, 0.0], np.pi + np.pi / 6)
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            cap_pose.pos, body_pose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)


class Centrifuge_5430(Centrifuge_Eppendorf_5430):

    def get_slot_pose(self, data: mujoco.MjData, slot_id: int) -> Pose:
        if slot_id < 0 or slot_id >= self.num_slots:
            raise ValueError(f'Invalid slot id {slot_id}')
        pos = data.site_xpos[self.slot_sites[slot_id]]
        mat = data.site_xmat[self.slot_sites[slot_id]]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(pos, quat)

    def get_tube_pose(self, data: mujoco.MjData, slot_id: int, mode="distal") -> Pose:
        slot_pose = self.get_slot_pose(data, slot_id)
        if mode == "distal":
            rel_pos= np.array([0.0, 0.0, 0.005])
        elif mode == "proximal":
            rel_pos = np.array([0.0, 0.0, -0.03])
        rel_quat = np.array([1.0, 0.0, 0.0, -1.0])
        # rel_quat = np.array([-0.183, 0.683, 0.683, 0.183])
        rel_quat /= np.linalg.norm(rel_quat)
        return mul_pose(p1=slot_pose, p2=Pose(rel_pos, rel_quat))
    
    def rotor_perturb(self):
        return np.random.uniform(-0.1, 0.1)


class InsertCentrifuge5430(Task):
    default_scene = SCENE_ROOT / "insert_centrifuge_5430.xml"
    default_task = "insert_centrifuge_5430"

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
        self.tube = CentrifugeTube(self.model, "1/", "1/")
        self.tube2 = CentrifugeTube(self.model, "2/", "2/")
        self.rack = GridSlot(self.model, 'rack/')

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        slot_id = np.random.randint(-4, 9) % 30

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        # Randomize the rotor position
        perturbation = self.instrument.rotor_perturb()
        self.data.qpos[self.instrument.rotor_qposadr] += perturbation
        mujoco.mj_kinematics(self.model, self.data)

        # Set tube positions
        tubepos = self.rack.get_position(self.data, 0, 0, '0')
        quat = self.tube.get_pose(self.data).quat
        self.tube.set_pose(self.data, Pose(tubepos, quat))
        self.tube2.set_pose(self.data, self.instrument.get_tube_pose(self.data, (slot_id + 15) % 30, 'proximal'))
        mujoco.mj_kinematics(self.model, self.data)

        self.slot_id = slot_id
        self.tar_tubepose = self.instrument.get_tube_pose(self.data, self.slot_id, 'distal')
        self.final_tar_tubepose = self.instrument.get_tube_pose(self.data, self.slot_id, 'proximal')

        self.task_info = {
            'prefix': 'Insert a second centrifuge tube into the slot that is symmetrically opposite to the currently placed tube',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        tube_height = self.tube.get_body_pose(self.data).pos
        tube_pos_2 = [self.tube.get_body_pose(self.data).pos[0], self.tube.get_body_pose(self.data).pos[1], self.tube.get_body_pose(self.data).pos[2]]
        site_pos = [self.final_tar_tubepose.pos[0], self.final_tar_tubepose.pos[1], self.final_tar_tubepose.pos[2]]
        squared_distance_site = sum((p1 - p2) ** 2 for p1, p2 in zip(tube_pos_2, site_pos))
        distance_site = math.sqrt(squared_distance_site)
        return 0.955 < tube_height[2] < 0.961 and distance_site < 0.005

class InsertCentrifuge5430Expert(InsertCentrifuge5430, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=0.8,
            qc_acc=0.6,
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
    
    def move_to(self, pose: Pose, num_steps: int=2):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)
    
    def gripper_control(self, value: float, delay: int=300):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(delay):
            self.step_and_log({})
    
    def wait(self, steps: int):
        for _ in range(steps):
            self.step_and_log({})

    @property
    def site_pose(self):
        return self.arm.get_site_pose(self.data)

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        path = self.interpolate(self.site_pose, self.tube.get_eef_pose(self.data), 10)
        self.path_follow(path)
        self.gripper_control(240)
        self.move_to(Pose(
            pos=self.site_pose.pos + (0.0, 0.0, 0.1),
            quat=self.site_pose.quat
        ), 20)
        tube_pose = self.tube.get_body_pose(self.data)
        rel_pose = mul_pose(p1=neg_pose(tube_pose), p2=self.site_pose)
        tar_pose = mul_pose(p1=self.tar_tubepose, p2=rel_pose)
        path = self.interpolate2(self.site_pose, tar_pose, 20)
        self.path_follow(path)
        # tube_pose = self.tube.get_body_pose(self.data)
        # rel_pose = mul_pose(p1=neg_pose(tube_pose), p2=self.site_pose)
        tar_pose = mul_pose(p1=self.final_tar_tubepose, p2=rel_pose)
        self.move_to(tar_pose, 20)
        self.gripper_control(190)
        self.wait(200)
        self.finish()

InsertCentrifuge5430.Expert = InsertCentrifuge5430Expert

if __name__ == "__main__":
    from tqdm import trange
    spec = InsertCentrifuge5430.load()
    expert = InsertCentrifuge5430.Expert(spec)
    for i in trange(100):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
