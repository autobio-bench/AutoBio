import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import Pose, slerp, IK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT

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

        self.dof = 6  
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof) 
        self.act_span = range(self.act_id, self.act_id + self.dof) 
        self.ik: IK = None  

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name) 

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]  
        quat = np.zeros(4)  
        mujoco.mju_mat2Quat(quat, mat)  
        return Pose(data.site_xpos[self.site_id], quat) 

    def qpos_perturb(self):
        lows = (-0.2, 0.0, 0.0, -0.05, 0.0, -0.1) 
        highs = (0.2, 0.3, 0.1, 0.05, 0.3,  0.1) 
        perturbation = np.random.uniform(lows, highs) 
        return perturbation  


class CentrifugeTube:

    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str):
        self.model = model
        self.cap_id = model.body(f'{cap_prefix}centrifuge_50ml_screw_cap').id
        self.cap_jnt_adr = model.joint(f'centrifuge_50ml_screw_cap').qposadr.item()
        self.body_id = model.body(f'{body_prefix}centrifuge_50ml_screw_body').id
        self.body_jnt_adr = model.joint(f'centrifuge_50ml_screw_body').qposadr.item()
        self.cap_pos_span = range(self.cap_jnt_adr, self.cap_jnt_adr + 3)
        self.body_pos_span = range(self.body_jnt_adr, self.body_jnt_adr + 3)
    
    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def get_end_effector_pose(self, data: mujoco.MjData, random: bool=False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.005])
        pos = cap_pos + bias
        rel_quat = np.zeros(4)
        rel_quat1 = np.zeros(4)
        quat1 = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi)    
        mujoco.mju_axisAngle2Quat(rel_quat1, np.array([0.0, 0.0, 1.0]), np.pi)   
        mujoco.mju_mulQuat(quat1, body_quat, rel_quat) 
        mujoco.mju_mulQuat(quat1, quat1, rel_quat1)

        return Pose(pos, quat1)


class Insert(Task):
    default_scene = SCENE_ROOT / "insert.xml"
    default_task = "insert"

    time_limit = 15.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        manager = Manager.from_spec(spec, [])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.rack1 = GridSlot(self.model, '1/')
        self.rack2 = GridSlot(self.model, '2/')
        self.object = CentrifugeTube(self.model, '1/', '2/')
        self.tube_end_pos = np.zeros(3)

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation

        # Randomize the tube position
        start_row = np.random.randint(2)
        start_col = np.random.randint(5)
        tube_pos = self.rack1.get_position(self.data, start_row, start_col, '50ml')
        self.data.qpos[self.object.body_pos_span] = tube_pos
        self.data.qpos[self.object.cap_pos_span] = tube_pos
        mujoco.mj_kinematics(self.model, self.data)

        # Randomize the tube end-position
        end_row = np.random.randint(2)
        end_col = np.random.randint(5)
        self.tube_end_pos = self.rack2.get_position(self.data, end_row, end_col, '50ml')

        self.task_info = {
            'prefix': f'pick up the centrifuge tube and move it to the other rack, row {end_row}, col {end_col}',
            'state_indices': list(range(7)),
            'action_indices': list(range(7)),
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        target_pos = [self.tube_end_pos[0], self.tube_end_pos[1]]
        tube_pos = self.object.get_body_pose(self.data).pos
        pos = [tube_pos[0], tube_pos[1]]
        distance = sum((p1 - p2) ** 2 for p1, p2 in zip(pos, target_pos)) ** 0.5
        return distance < 0.015 and 0.829 < tube_pos[2] < 0.831

class InsertExpert(Insert, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=0.8,
            qc_acc=0.8,
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

    def gripper_control(self, value: float):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(200):
            self.step_and_log({})
    
    def wait(self, steps: int):
        for _ in range(steps):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        # Pick up the tube
        self.gripper_control(0)
        eef_pose = self.object.get_end_effector_pose(self.data)
        eef_pre_pose = Pose(pos=eef_pose.pos + (0.0, 0.0, 0.04), quat=eef_pose.quat)
        cur_pose = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pose, eef_pre_pose, 5)
        path2 = self.interpolate(eef_pre_pose, eef_pose, 10)
        self.path_follow(path)
        self.path_follow(path2)
        self.gripper_control(240)

        cur_pose = self.arm.get_site_pose(self.data)
        height = cur_pose.pos[2]
        terminal_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, terminal_pose, 20)
        self.path_follow(path)

        # Transfer the tube


        cur_pose = self.arm.get_site_pose(self.data)
        pre_end_pos = self.tube_end_pos.copy()
        pre_end_pos[2] = cur_pose.pos[2]
        pre_end_pose = Pose(pos=pre_end_pos, quat=cur_pose.quat)
        path = self.interpolate(cur_pose, pre_end_pose, 20)
        self.path_follow(path)

        cur_pose = self.arm.get_site_pose(self.data)
        end_pos = cur_pose.pos.copy()
        end_pos[2] = height + 0.001
        end_pose = Pose(pos=end_pos, quat=cur_pose.quat)
        path = self.interpolate(pre_end_pose, end_pose, 20)
        self.path_follow(path)
        self.gripper_control(0)

        # Final adjustments
        cur_pose = self.arm.get_site_pose(self.data)
        terminal_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, terminal_pose, 20)
        self.path_follow(path)

        self.finish()

Insert.Expert = InsertExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = Insert.load()
    expert = Insert.Expert(spec)
    for i in trange(100):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
