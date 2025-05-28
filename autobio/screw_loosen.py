import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import IK, Pose, slerp, AlohaAnalyticalIK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

class AlohaArm:

    # 6-DOF arm

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}left/waist'
        self.act_name = f'{prefix}left/waist'
        self.site_name = f'{prefix}left/gripper'
        self.base_name = f'{prefix}left/base_link'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_jnt_adr = model.joint(f'{prefix}left/right_finger').qposadr.item()
        self.gripper_id = model.actuator(f'{prefix}left/gripper').id
        self.nv = model.nv
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        parentid = self.model.body(self.base_name).parentid.item()
        pose = Pose(data.xpos[parentid], data.xquat[parentid])
        self.ik = AlohaAnalyticalIK(pose=pose)
    
    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)
    
    def get_jac(self, data: mujoco.MjData) -> np.ndarray:
        jac = np.zeros((6, self.nv))
        mujoco.mj_jacSite(self.model, data, jac[:3], jac[3:], self.site_id)
        jac = jac[:, self.jnt_adr: self.jnt_adr + self.dof]
        return jac

    def qpos_perturb(self):
        lows = (-0.1, 0.0, 0.0, -0.025, 0.0, -0.05)
        highs = (0.1, 0.15, 0.05, 0.025, 0.15,  0.05)
        perturbation = np.random.uniform(lows, highs)
        return perturbation
    
class CentrifugeTube:

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.cap_id_50 = model.body('6/centrifuge_50ml_screw_cap').id
        self.cap_jnt_adr_50 = model.joint('centrifuge_50ml_screw_cap').qposadr.item()
        self.body_id_50 = model.body('7/centrifuge_50ml_screw_body').id
        self.body_jnt_adr_50 = model.joint('centrifuge_50ml_screw_body').qposadr.item()
        # self.cap_id_15 = model.body('5/centrifuge_15ml_cap').id
        # self.cap_jnt_adr_15 = model.joint('centrifuge_15ml_cap').qposadr.item()
        # self.body_id_15 = model.body('4/centrifuge_15ml_body').id
        # self.body_jnt_adr_15 = model.joint('centrifuge_15ml').qposadr.item()
        self.cap_id = self.cap_id_50
        self.cap_jnt_adr = self.cap_jnt_adr_50
        self.body_id = self.body_id_50
        self.body_jnt_adr = self.body_jnt_adr_50
        self.cap_pos_span = range(self.cap_jnt_adr, self.cap_jnt_adr + 3)
        self.body_pos_span = range(self.body_jnt_adr, self.body_jnt_adr + 3)
        self.tube_params = {
            '50ml': {
                'rows': 5,
                'columns': 2,
                'x0': -0.072,
                'y0': -0.018,
                'delta_x': 0.036,
                'delta_y': 0.036
            }
            # ,
            # '15ml': {
            #     'rows': 6,
            #     'columns': 3,
            #     'x0': -0.09,
            #     'y0': -0.036,
            #     'delta_x': 0.036,
            #     'delta_y': 0.036
            # }
        }
    
    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def randomposition(self, data: mujoco.MjData):
        selected_type = np.random.choice(list(self.tube_params.keys()))
        params = self.tube_params[selected_type]
        # if(selected_type == "15ml"):
        #     self.cap_id = self.cap_id_15
        #     self.cap_jnt_adr = self.cap_jnt_adr_15
        #     self.body_id = self.body_id_15
        #     self.body_jnt_adr = self.body_jnt_adr_15
        # else:
        self.cap_id = self.cap_id_50
        self.cap_jnt_adr = self.cap_jnt_adr_50
        self.body_id = self.body_id_50
        self.body_jnt_adr = self.body_jnt_adr_50
        self.cap_pos_span = range(self.cap_jnt_adr, self.cap_jnt_adr + 3)
        self.body_pos_span = range(self.body_jnt_adr, self.body_jnt_adr + 3)
        row_idx = np.random.randint(0, params['rows'])
        col_idx = np.random.randint(0, params['columns'])
        x = params['x0'] + row_idx * params['delta_x']
        y = params['y0'] + col_idx * params['delta_y']
        self._set_joint_positions(selected_type, x, y, data)
        return selected_type

    def _set_joint_positions(self, tube_type, x, y, data: mujoco.MjData):
        if tube_type == "50ml":
            data.joint("centrifuge_50ml_screw_body").qpos[0] = x
            data.joint("centrifuge_50ml_screw_body").qpos[1] = y
            data.joint("centrifuge_50ml_screw_cap").qpos[0] = x
            data.joint("centrifuge_50ml_screw_cap").qpos[1] = y
        #     data.joint("centrifuge_15ml").qpos[0] = -0.3
        #     data.joint("centrifuge_15ml").qpos[1] = -0.018
        #     data.joint("centrifuge_15ml_cap").qpos[0] = -0.3
        #     data.joint("centrifuge_15ml_cap").qpos[1] = -0.018
        # elif tube_type == "15ml":
        #     data.joint("centrifuge_15ml").qpos[0] = x
        #     data.joint("centrifuge_15ml").qpos[1] = y
        #     data.joint("centrifuge_15ml_cap").qpos[0] = x
        #     data.joint("centrifuge_15ml_cap").qpos[1] = y
        #     data.joint("centrifuge_50ml_screw_body").qpos[0] = -0.318
        #     data.joint("centrifuge_50ml_screw_body").qpos[1] = 0.0
        #     data.joint("centrifuge_50ml_screw_cap").qpos[0] = -0.318
        #     data.joint("centrifuge_50ml_screw_cap").qpos[1] = 0.0

    @staticmethod
    def randomsphere(x, y, z, d):
        radius = d / 2.0
        while True:
            x_dir, y_dir, z_dir = np.random.normal(0, 1, size=3)
            norm = np.linalg.norm([x_dir, y_dir, z_dir])
            if norm > 0:
                break
        x_unit = x_dir / norm
        y_unit = y_dir / norm
        z_unit = z_dir / norm
        u = np.random.uniform(0, 1)
        d = radius * (u ** (1/3))
        x = x + x_unit * d
        y = y + y_unit * d
        z = z + z_unit * d
        return np.array([x, y, z])
    
class ScrewLoose(Task):
    default_scene = SCENE_ROOT / "lab_screw_all.xml"
    default_task = "screw_loose"

    time_limit = 41.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body_name1 = f'1/aloha:left/base_link'
        body_name2 = f'2/aloha:left/base_link'
        set_gravcomp(spec.body(body_name1))
        set_gravcomp(spec.body(body_name2))
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        manager = Manager.from_spec(spec, [])
        super().__init__(manager)
        self.arm1 = AlohaArm(self.model, '1/aloha:')
        self.arm2 = AlohaArm(self.model, '2/aloha:')
        self.object = CentrifugeTube(self.model)

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize arm joint positions
        perturbation = self.arm1.qpos_perturb()
        self.data.qpos[self.arm1.jnt_span] += perturbation
        self.data.ctrl[self.arm1.act_span] += perturbation
        perturbation = self.arm2.qpos_perturb()
        self.data.qpos[self.arm2.jnt_span] += perturbation
        self.data.ctrl[self.arm2.act_span] += perturbation

        # Randomize tube position
        tubetype = self.object.randomposition(self.data)

        self.task_info = {
            'prefix': 'dual-Aloha arms unscrewing centrifuge tube cap: one stabilizes tube while the other twists cap',
            'state_indices': self.arm1.state_indices + self.arm2.state_indices,
            'action_indices': self.arm1.action_indices + self.arm2.action_indices,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '1/aloha:wrist_cam_left',
                'wrist_image_2': '2/aloha:wrist_cam_left'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        height = self.object.get_cap_pose(self.data).pos[2] - self.object.get_body_pose(self.data).pos[2]
        geom1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "1/aloha:leftfinger")
        geom2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "6/cap_cyl")
        geom3_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "6/nut")
        geom4_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "7/bolt")
        geom_is_contact = 0
        tube_is_not_contact = 1
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (con.geom1 == geom2_id and con.geom2 == geom1_id):
                geom_is_contact = 1
            if (con.geom1 == geom3_id and con.geom2 == geom4_id) or (con.geom1 == geom4_id and con.geom2 == geom3_id):
                tube_is_not_contact = 0
        return height > 0.15 and geom_is_contact == 1 and tube_is_not_contact == 1

class ScrewLooseExpert(ScrewLoose, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm1.register_ik(self.data)
        self.arm2.register_ik(self.data)
        self.planner_1 = Topp(
            dof=self.arm1.dof,
            qc_vel=3.0,
            qc_acc=2.5,
            ik=self.arm1.ik.solve,
        )
        self.planner_2 = Topp(
            dof=self.arm2.dof,
            qc_vel=3.0,
            qc_acc=2.5,
            ik=self.arm2.ik.solve,
        )

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def path_follow(self, path: list[Pose], arm: AlohaArm):
        self.planner_1.ik = arm.ik.solve
        trajectory = self.planner_1.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = self.planner_1.query(trajectory, t)
                self.data.ctrl[arm.act_span] = ctrl
            self.step_and_log({})

    def path_follow_dual(self, path_1: list[Pose], path_2: list[Pose], arm_1: AlohaArm, arm_2: AlohaArm):
        trajectory_1 = self.planner_1.jnt_traj(path_1)
        trajectory_2 = self.planner_2.jnt_traj(path_2)
        run_time = max(trajectory_1.duration, trajectory_2.duration) + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl_1 = self.planner_1.query(trajectory_1, t)
                ctrl_2 = self.planner_2.query(trajectory_2, t)
                self.data.ctrl[arm_1.act_span] = ctrl_1
                self.data.ctrl[arm_2.act_span] = ctrl_2
            self.step_and_log({})

    def gripper_control(self, value: float, arm: AlohaArm):
        self.data.ctrl[arm.gripper_id] = value
        for _ in range(200):
            self.step_and_log({})

    def position_track(self, arm: AlohaArm, target_position, num_steps):
        current_position = arm.get_site_pose(self.data)
        path = self.interpolate(current_position, target_position, num_steps)
        self.path_follow(path, arm)

    def cap_loose(self, gripper_value, loops, arm1: AlohaArm, mid1_pos: Pose):
        for j in range(loops):
            if j == 0:
                inner_loops = 4
            else:
                inner_loops = 8
            self.gripper_control(gripper_value, arm1)
            current_position = self.arm1.get_site_pose(self.data)
            now_pos = current_position
            now_path = []
            for i in range(inner_loops):
                ang = -np.pi / 10
                delta_quat = np.array([np.cos(ang/2), np.sin(ang/2), 0., 0.])
                new_quat = np.zeros(4)
                mujoco.mju_mulQuat(new_quat, now_pos.quat, delta_quat)
                next_pos = Pose(pos=now_pos.pos + (0.0, 0.0, 0.003 * 1 / 1000), quat=new_quat)
                next_path = self.interpolate(now_pos, next_pos, 20)
                now_path.extend(next_path[1:])
                now_pos = next_pos
            self.path_follow(now_path, arm1)
            if j == 3:
                tube_pos = Pose(pos=mid1_pos.pos + (0.0, 0.0, 0.02), quat=now_pos.quat)
                self.position_track(self.arm1, tube_pos, 5)
            else:
                mid1_pos.quat = np.array([0.560977785, 0.430454196, 0.560977785, -0.430454196])
                mid2_pos = Pose(pos=next_pos.pos, quat=mid1_pos.quat)
                self.gripper_control(0.037, arm1)
                self.position_track(arm1, mid1_pos, 5)
                self.position_track(arm1, mid2_pos, 5)

    def execute(self):
        self.arm1.ik.initial_qpos = self.data.qpos[self.arm1.jnt_span]
        self.arm2.ik.initial_qpos = self.data.qpos[self.arm2.jnt_span]
        gripper_val = 0.02
        self.gripper_control(0.030, self.arm1)
        self.gripper_control(0.030, self.arm2)
        target_quat1 = np.array([0.7071, 0., 0.7071, 0.])
        target_quat1 /= np.linalg.norm(target_quat1)
        target_quat2 = np.array([0.7071, 0., 0., 0.7071])
        target_quat2 /= np.linalg.norm(target_quat2)
        tube_pos = self.object.get_body_pose(self.data)
        arm1_pose = self.arm1.get_site_pose(self.data)
        arm2_pose = self.arm2.get_site_pose(self.data)
        tube_pos_random = Pose(pos=self.object.randomsphere(0.0, 0.0, 1.05, 0.05),quat=target_quat1)

        tube_pos_mid1 = Pose(pos=tube_pos_random.pos + (0.0, -0.1, -0.055), quat=target_quat2)
        tube_pos_mid2 = Pose(pos=tube_pos.pos + (0.0, 0.0, 0.14), quat=target_quat1)
        tube_pos_mid3 = Pose(pos=tube_pos_mid2.pos + (0.0, 0.0, -0.015), quat=target_quat1)
        path_1_1 = self.interpolate(arm1_pose, tube_pos_mid2, 5)
        path_2_1 = self.interpolate(arm2_pose, tube_pos_mid1, 5)
        path_1_2 = self.interpolate(tube_pos_mid2, tube_pos_mid3, 5)
        path_1_1.extend(path_1_2[1:])
        self.path_follow_dual(path_1_1, path_2_1, self.arm1, self.arm2)
        self.gripper_control(gripper_val, self.arm1)

        tube_pos_mid4 = Pose(pos=self.arm1.get_site_pose(self.data).pos + (0.0, 0.0, 0.1), quat=target_quat1)
        tube_pos_mid5 = Pose(pos=tube_pos_random.pos + (0.0, 0.0, -0.055), quat=target_quat2)
        path_1_3 = self.interpolate(self.arm1.get_site_pose(self.data), tube_pos_mid4, 5)
        path_1_4 = self.interpolate(tube_pos_mid4, tube_pos_random, 5)
        path_1_3.extend(path_1_4[1:])
        self.path_follow(path_1_3, self.arm1)
        self.position_track(self.arm2, tube_pos_mid5, 5)
        self.gripper_control(0.005, self.arm2)

        tube_pos_mid6 = Pose(pos=tube_pos_random.pos + (0.0, 0.0, 0.01), quat=target_quat1)
        self.cap_loose(gripper_val, 4, self.arm1, tube_pos_mid6)

        self.serializer.finish()

ScrewLoose.Expert = ScrewLooseExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = ScrewLoose.load()
    expert = ScrewLoose.Expert(spec)
    for i in trange(1):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
