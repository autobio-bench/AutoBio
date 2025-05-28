import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import IK, Pose, slerp, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import Thermal_mixer_eppendorf_c, UIStateCoordinator
from thermal_mixer_ui import Time

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

def compose_quaternions(*qs):
    result = np.array([1.0, 0.0, 0.0, 0.0])
    for q in reversed(qs):
        mujoco.mju_mulQuat(result, result, q)
    return result

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
    
    def random_row_col(self, slot_type: str = 'default') -> tuple:
        grid = self.grids[slot_type]
        row = np.random.randint(grid['rows'])
        col = np.random.randint(grid['cols'])
        return row, col

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

class ThermalBlock:
    def __init__(self, model: mujoco.MjModel, prefix: str = "thermoblock-function"):
        self.model = model
        self.prefix = prefix 
        self.grids = dict()
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if not site_name.startswith(prefix):
                continue
            parts = site_name.split('-')
            slot_type = parts[1] if len(parts) > 1 else 'default'
            if slot_type not in self.grids:
                info = dict()
                info['origin'] = i
                user = self.model.site_user[i]
                info['rows'] = int(user[1])
                info['row_gap'] = float(user[2])
                info['cols'] = int(user[3])
                info['col_gap'] = float(user[4])
                info['height'] = float(user[5])
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

    def random_row_col(self, slot_type: str = 'default') -> tuple:
        grid = self.grids[slot_type]
        row = np.random.randint(grid['rows'])
        col = np.random.randint(grid['cols'])
        return row, col

    def set_insertion_point(self, data: mujoco.MjData, row: int, col: int, slot_type: str='default', hei: int=0):
        pos = self.get_position(data, row, col, slot_type, hei)
        site_id = self.grids[slot_type]['origin']
        data.site_xpos[site_id] = pos
        return pos
    
class CentrifugeTube:
    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str, prefix: str = 'centrifuge_1-5ml_screw'):
        self.model = model
        self.prefix = prefix
        self.cap_id = model.body(f'{cap_prefix}centrifuge_1-5ml_screw_cap').id
        self.body_id = model.body(f'{body_prefix}centrifuge_1-5ml_screw_body').id
        self.joint_id = model.joint('centrifuge_1-5ml_screw_joint').qposadr.item()
        self.pos_span = range(self.joint_id, self.joint_id + 3)
        self.quat_span = range(self.joint_id + 3, self.joint_id + 7)
    
    def get_pose(self, data: mujoco.MjData) -> Pose:
        pos = data.qpos[self.pos_span]
        quat = data.qpos[self.quat_span]
        return Pose(pos, quat)

    def set_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span] = pose.pos
        data.qpos[self.quat_span] = pose.quat
        print(f"Set tube position to: {pose.pos}, quaternion: {pose.quat}")

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def get_end_effector_pose(self, data: mujoco.MjData, random: bool=False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.005])
        pos = cap_pos + bias
        quat = np.zeros(4)
        quat_rel = np.zeros(4)
        axis = np.array([0.0, 1.0, 0.0])
        mujoco.mju_axisAngle2Quat(quat_rel, axis, np.pi / 2)
        mujoco.mju_mulQuat(quat, quat_rel, body_quat)
        trans_quat = compose_quaternions(
            np.array([0.7071, 0.0, 0.0, 0.7071]),
            np.array([0.7071, 0.0, 0.7071, 0.0]),
            quat
        )
        return Pose(pos, trans_quat)

class Thermal_mixer(Thermal_mixer_eppendorf_c):

    def _reset(self, data):
        super()._reset(data)
    
    def get_eefpose_lever(self, sitepose: Pose, mode: str='detach') -> Pose:
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 1.0, 0.0]), -np.pi/6)
        match mode:
            case 'detach':
                rel_pos = np.array([0.0, 0.0, -0.05])
            case 'adjust':
                rel_pos = np.array([0.0018, 0.0, -0.01])
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

    def get_eef_pose(self, data: mujoco.MjData, buttonID: str, mode: str='detach'):
        match buttonID:
            case 'time_up':
                site_pos = data.site_xpos[self.time_up_site]
                site_mat = data.site_xmat[self.time_up_site]
            case 'time_down':
                site_pos = data.site_xpos[self.time_down_site]
                site_mat = data.site_xmat[self.time_down_site]
            case 'temp_up':
                site_pos = data.site_xpos[self.temp_up_site]
                site_mat = data.site_xmat[self.temp_up_site]
            case 'temp_down':
                site_pos = data.site_xpos[self.temp_down_site]
                site_mat = data.site_xmat[self.temp_down_site]
            case 'freq_up':
                site_pos = data.site_xpos[self.speed_up_site]
                site_mat = data.site_xmat[self.speed_up_site]
            case 'freq_down':
                site_pos = data.site_xpos[self.speed_down_site]
                site_mat = data.site_xmat[self.speed_down_site]
            case 'none':
                return None
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, site_mat)
        return self.get_eefpose_lever(Pose(site_pos, quat), mode)
    
    def generate_random_parameters(self):
        def rand(low, high, exception):
            while True:
                value = np.random.randint(low, high)
                if value != exception:
                    return value
            
        time_step = rand(-10, 11, 0)
        time_obj = Time(seconds=60)
        time_sign = 1 if time_step > 0 else -1
        time_step = abs(time_step)
        for i in range(time_step):
            time_obj.seconds += time_sign * time_obj.step_size
        set_time = time_obj.seconds
        set_temp = 25 + rand(-15, 16, 0)
        set_rpms = 900 + rand(-12, 13, 0) * 50
        return np.array([set_time, set_temp, set_rpms])

class ThermalMixerManipulate(Task):
    default_scene = SCENE_ROOT / "mani_thermal_mixer.xml"
    default_task = "thermal_mixer"

    time_limit = 30.0
    early_stop = False

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = Thermal_mixer('/thermal_mixer_eppendorf_c:')
        ui_state_coordinator = UIStateCoordinator()
        manager = Manager.from_spec(spec, [self.instrument, ui_state_coordinator])
        super().__init__(manager)
        # self.rack = GridSlot(self.model, 'rackcentrifuge_plate-')
        # self.object = CentrifugeTube(self.model, 'tubecap', 'tubebody')
        self.thermoblock = ThermalBlock(self.model, '/thermal_mixer_eppendorf_c:thermoblock-function')
        self.arm = UR5eArm(self.model, '/ur:')

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        # row = np.random.randint(5)
        # col = np.random.randint(12)
        # tube_pos = self.rack.get_position(self.data, row, col, slot_type='0')
        # self.data.qpos[self.object.pos_span] = tube_pos
        mujoco.mj_kinematics(self.model, self.data)
        # self.tube_pos = tube_pos 
        # self.tube_quat = np.array([1, 0, 0, 0])

        # Generate random parameters for the instrument
        set_time, set_temp, set_rpm = self.set_params = self.instrument.generate_random_parameters()

        # inst_row, inst_col = self.thermoblock.random_row_col('function')
        # self.inst_row = inst_row
        # self.inst_col = inst_col
        # self.inst_target_pos = self.thermoblock.set_insertion_point(self.data, inst_row, inst_col, slot_type='function')

        self.task_info = {
            'prefix': f'Adjust thermal mixer parameters, with speed set to {set_rpm} RPM, temperature set to {set_temp} °C, and time set to {set_time} seconds',
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
        initial = {
            'time': 60,
            'set_temperature': 25.0,
            'frequency': 900
        }
        target = {
            'time': self.set_params[0],
            'set_temperature': self.set_params[1],
            'frequency': self.set_params[2]
        }
        current = {
            'time': self.instrument.ui_state.main_parameter.time.seconds,
            'set_temperature': self.instrument.ui_state.main_parameter.set_temperature,
            'frequency': self.instrument.ui_state.main_parameter.frequency
        }
        weights = {
            'set_temperature': 0.5,
            'time': 0.3,
            'frequency': 0.2
        }
        scores = {}
        temp_range = abs(target['set_temperature'] - initial['set_temperature'])
        if temp_range > 0:
            temp_diff = abs(current['set_temperature'] - target['set_temperature'])
            scores['set_temperature'] = max(0, 1 - (temp_diff / temp_range))
        else:
            scores['set_temperature'] = 1.0

        time_range = abs(target['time'] - initial['time'])
        if time_range > 0:
            time_diff = abs(current['time'] - target['time'])
            scores['time'] = max(0, 1 - (time_diff / time_range))
        else:
            scores['time'] = 1.0
        
        freq_range = abs(target['frequency'] - initial['frequency'])
        if freq_range > 0:
            freq_diff = abs(current['frequency'] - target['frequency'])
            scores['frequency'] = max(0, 1 - (freq_diff / freq_range))
        else:
            scores['frequency'] = 1.0
        
        total_score = 0.0
        for param, weight in weights.items():
            total_score += scores[param] * weight

        return float(min(1, max(0, total_score)))


class ThermalMixerManipulateExpert(ThermalMixerManipulate, Expert):
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

    def gripper_control(self, value: float):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(300):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        set_params = self.set_params
        if self.task == 'thermal_mixer':
            actual_time = self.instrument.ui_state.main_parameter.time.seconds
            actual_temp = self.instrument.ui_state.main_parameter.set_temperature
            actual_freq = self.instrument.ui_state.main_parameter.frequency
            if set_params[0] > actual_time:
                time_button_ID="time_up"
            elif set_params[0] < actual_time:
                time_button_ID="time_down"
            else:
                time_button_ID="none"
            time_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=time_button_ID)
            if set_params[1] > actual_temp:
                temp_button_ID="temp_up"
            elif set_params[1] < actual_temp:
                temp_button_ID="temp_down"
            else:
                temp_button_ID="none"
            temp_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=temp_button_ID)
            if set_params[2] > actual_freq:
                freq_button_ID = "freq_up"
                freq_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID="freq_up")
            elif set_params[2] < actual_freq:
                freq_button_ID = "freq_down"
            else:
                freq_button_ID = "none"
            freq_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=freq_button_ID)
            self.gripper_control(255)

            if freq_pose is not None:
                self.move_to(freq_pose, num_steps=2)
                freq_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=freq_button_ID)
                self.move_to(freq_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.frequency != set_params[2]:
                    self.step_and_log({})
                self.move_to(freq_pose, num_steps=2)

            if temp_pose is not None:
                self.move_to(temp_pose, num_steps=2)
                temp_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=temp_button_ID)
                self.move_to(temp_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.set_temperature != set_params[1]:
                    self.step_and_log({})
                self.move_to(temp_pose, num_steps=2)
            
            if time_pose is not None:
                self.move_to(time_pose, num_steps=2)
                time_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=time_button_ID)
                self.move_to(time_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.time.seconds != set_params[0]:
                    self.step_and_log({})
                self.move_to(time_pose, num_steps=2)
        elif self.task == 'thermal_mixer_comprehensive':
            eef_pose = self.object.get_end_effector_pose(self.data)
            eef_pre_pose = Pose(pos=eef_pose.pos + (0.0, 0.0, 0.05), quat=eef_pose.quat)
            eef_pose = self.object.get_end_effector_pose(self.data)
            cur_pose = self.arm.get_site_pose(self.data)
            path = self.interpolate(cur_pose, eef_pre_pose, 5)
            path_ = self.interpolate(eef_pre_pose, eef_pose, 10)
            path.extend(path_[1:])
            self.path_follow(path)
            self.gripper_control(240)
            
            cur_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(pos=cur_pose.pos + np.array([0.0, 0.0, 0.1]), quat=cur_pose.quat)
            path_lift = self.interpolate(cur_pose, lift_pose, 20)
            self.path_follow(path_lift)

            tube_quat = self.object.get_end_effector_pose(self.data).quat
            target_pos = self.inst_target_pos
            pre_target_pose = Pose(pos=target_pos + np.array([0.0, 0.0, 0.28]), quat=tube_quat)
            path_to_target = self.interpolate(lift_pose, pre_target_pose, 5)
            self.path_follow(path_to_target)

            cur_pose = self.arm.get_site_pose(self.data)
            path_insert_tube = self.interpolate(pre_target_pose, Pose(pos=target_pos+ np.array([0.0, 0.0, 0.025]), quat=cur_pose.quat), 5)
            self.path_follow(path_insert_tube)
            self.gripper_control(100)

            cur_pose = self.arm.get_site_pose(self.data)
            cur_pose.pos[2] += 0.01
            lift_pose = Pose(pos=cur_pose.pos + np.array([0.0, 0.0, 0.05]), quat=cur_pose.quat)
            path_lift = self.interpolate(cur_pose, lift_pose, 10)
            self.path_follow(path_lift)

            actual_time = self.instrument.ui_state.main_parameter.time.seconds
            actual_temp = self.instrument.ui_state.main_parameter.set_temperature
            actual_freq = self.instrument.ui_state.main_parameter.frequency
            set_params = self.instrument.generate_random_parameters()
            if set_params[0] > actual_time:
                time_button_ID="time_up"
            elif set_params[0] < actual_time:
                time_button_ID="time_down"
            else:
                time_button_ID="none"
            time_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=time_button_ID)
            if set_params[1] > actual_temp:
                temp_button_ID="temp_up"
            elif set_params[1] < actual_temp:
                temp_button_ID="temp_down"
            else:
                temp_button_ID="none"
            temp_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=temp_button_ID)
            if set_params[2] > actual_freq:
                freq_button_ID = "freq_up"
                freq_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID="freq_up")
            elif set_params[2] < actual_freq:
                freq_button_ID = "freq_down"
            else:
                freq_button_ID = "none"
            freq_pose = self.instrument.get_eef_pose(self.data, mode='detach', buttonID=freq_button_ID)
            self.gripper_control(255)

            if freq_pose is not None:
                self.move_to(freq_pose, num_steps=2)
                freq_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=freq_button_ID)
                self.move_to(freq_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.frequency != set_params[2]:
                    self.step_and_log({})
                self.move_to(freq_pose, num_steps=2)

            if temp_pose is not None:
                self.move_to(temp_pose, num_steps=2)
                temp_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=temp_button_ID)
                self.move_to(temp_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.set_temperature != set_params[1]:
                    self.step_and_log({})
                self.move_to(temp_pose, num_steps=2)
            
            if time_pose is not None:
                self.move_to(time_pose, num_steps=2)
                time_pose_press = self.instrument.get_eef_pose(self.data, mode='adjust', buttonID=time_button_ID)
                self.move_to(time_pose_press, num_steps=2)
                while self.instrument.ui_state.main_parameter.time.seconds != set_params[0]:
                    self.step_and_log({})
                self.move_to(time_pose, num_steps=2)
        self.serializer.finish()
        with self.serializer.within_save_dir():
            self.manager.finish()

ThermalMixerManipulate.Expert = ThermalMixerManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = ThermalMixerManipulate.load()
    expert = ThermalMixerManipulate.Expert(spec)
    for i in trange(100):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
