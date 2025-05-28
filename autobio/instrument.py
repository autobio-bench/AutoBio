from copy import deepcopy

import numpy as np
import mujoco

from simulation import System
from thermal_mixer_ui import NoProgramMain, StatusBar, MainParameter, Time

class FlatButton(System):
    def _reload(self, _):
        self.sensor_id = self.name2id(mujoco.mjtObj.mjOBJ_SENSOR)
        self.sensor_adr = self.model.sensor_adr[self.sensor_id].item()
        self.dt = self.model.opt.timestep
    
    def _reset(self, data: mujoco.MjData):
        self.value = 0.0
        self.pressed_time = 0.0
        self.is_pressed = False
        self._update(data)
    
    def _update(self, data: mujoco.MjData):
        self.value = data.sensordata[self.sensor_adr]
        if self.value > 0.5:
            self.pressed_time += self.dt
            self.is_pressed = True
        else:
            self.pressed_time = 0.0
            self.is_pressed = False
    
    def reset_pressed_time(self):
        self.pressed_time = 0.0


class Detent(System):
    def _reload(self, _):
        self.joint_id = self.name2id(mujoco.mjtObj.mjOBJ_JOINT)
        self.qposadr = self.model.jnt_qposadr[self.joint_id].item()
        self.joint_low = self.model.jnt_range[self.joint_id, 0].item()
        self.joint_high = self.model.jnt_range[self.joint_id, 1].item()
        self.joint_num_steps = self.model.jnt_user[self.joint_id, 0].item()
        self.joint_step = (self.joint_high - self.joint_low) / (self.joint_num_steps - 1)

    def _reset(self, data: mujoco.MjData):
        self._update(data)  # Delegate to _update

    def _update(self, data: mujoco.MjData):
        qpos = data.qpos[self.qposadr]
        value = np.rint((qpos - self.joint_low) / self.joint_step).astype(int)
        value = np.clip(value, 0, self.joint_num_steps - 1)
        self.value = value

class VortexMixerGenie2(System):
    def _configure(self):
        self.switch = self.add_subsystem(Detent('switch'))
        self.knob = self.add_subsystem(Detent('knob'))

    def _reload(self, model: mujoco.MjModel):
        self.platform_actuator_id = self.name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, 'platform/pivot')
        self.platform_body_id = self.name2id(mujoco.mjtObj.mjOBJ_BODY, 'platform')
        geomnum = model.body_geomnum[self.platform_body_id].item()
        geomadr = model.body_geomadr[self.platform_body_id].item()
        assert geomnum > 0 and geomadr != -1
        self.platform_geom_id_range = range(geomadr, geomadr + geomnum)

    def _update(self, data: mujoco.MjData):
        switch = self.switch.value
        knob = self.knob.value
        if knob == 0:
            rpm = 0
        else:
            # 1 ~ 10 -> 600 ~ 3200 RPM
            rpm = np.interp(knob, [1, 10], [600, 3200])
        radps = rpm * 2 * np.pi / 60
        if switch == 0:
            # touch
            data.ctrl[self.platform_actuator_id] = radps if self._touch_active(data) else 0
        elif switch == 1:
            # off
            data.ctrl[self.platform_actuator_id] = 0
        elif switch == 2:
            # on
            data.ctrl[self.platform_actuator_id] = radps
        else:
            raise ValueError(f'Unknown switch state {switch}')

    def _touch_active(self, data: mujoco.MjData):
        # TODO: consider force
        for contact in data.contact:
            if contact.geom1 in self.platform_geom_id_range or contact.geom2 in self.platform_geom_id_range:
                return True
        return False

class ThermalCyclerBioradC1000(System):
    def _configure(self):
        self.lid_lever = self.add_subsystem(Detent('lid-lever'))
    
    def _reload(self, model: mujoco.MjModel):
        self.lid_lock = self.name2id(mujoco.mjtObj.mjOBJ_EQUALITY, 'lid-lock')
        self.lid_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid')
        self.lid_qpos_min = model.jnt_range[self.lid_joint, 0].item()
        self.lid_qposadr = model.jnt_qposadr[self.lid_joint].item()
        self.lid_force_knob_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid-force-knob')
        self.lid_force_knob_qposadr = model.jnt_qposadr[self.lid_force_knob_joint].item()
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
        self.lever_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid-lever')
        self.lever_qposadr = model.jnt_qposadr[self.lever_joint].item()
        self.lever_jntlimit = model.jnt_range[self.lever_joint]
        self.lever_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'lid-lever')
        self.knob_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'lid-force-knob')

    def _reset(self, data: mujoco.MjData):
        self._bad_locking = False
        self._update(data)  # Delegate to _update
    
    def _update(self, data):
        if self.lid_lever.value == 1:
            # unconditionally unlock
            self._bad_locking = False
            data.eq_active[self.lid_lock] = 0
        else:
            lid_qpos = data.qpos[self.lid_qposadr]
            if lid_qpos > self.lid_qpos_min + 0.01:
                # try to lock while lid is open
                self._bad_locking = True
            elif not self._bad_locking:
                # lock when lid is closed
                data.eq_active[self.lid_lock] = 1

class Centrifuge_Eppendorf_5430(System):
    
    def _reload(self, model: mujoco.MjModel):
        self.lid_lock = self.name2id(mujoco.mjtObj.mjOBJ_EQUALITY, 'lid-lock')
        self.lid_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid')
        self.lid_qpos_min = model.jnt_range[self.lid_joint, 0].item()
        self.lid_qposadr = model.jnt_qposadr[self.lid_joint].item()
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
        self.rotor_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'rotor')
        self.rotor_qposadr = model.jnt_qposadr[self.rotor_joint].item()
        self.lid_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'lid')
        self.slot_sites = [
            self.name2id(mujoco.mjtObj.mjOBJ_SITE, f'slot{i:02}') for i in range(30)
        ]
        self.num_slots = len(self.slot_sites)

    def _reset(self, data: mujoco.MjData):
        self._bad_locking = False
        self._update(data)  # Delegate to _update
    
    def _update(self, data):
        lid_qpos = data.qpos[self.lid_qposadr]
        if lid_qpos > self.lid_qpos_min + 0.01:
            # try to lock while lid is open
            self._bad_locking = True
        else:
            self._bad_locking = False
        if not self._bad_locking:
            # lock when lid is closed
            data.eq_active[self.lid_lock] = 1

class Centrifuge_Eppendorf_5910(System):
    
    def _reload(self, model: mujoco.MjModel):
        self.lid_lock = self.name2id(mujoco.mjtObj.mjOBJ_EQUALITY, 'lid-lock')
        self.lid_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid')
        self.lid_qpos_max = model.jnt_range[self.lid_joint, 1].item()
        self.lid_qposadr = model.jnt_qposadr[self.lid_joint].item()
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
        self.rotor_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'rotor-body')
        self.lid_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'lid')
        self.base_body = self.name2id(mujoco.mjtObj.mjOBJ_BODY, 'world')

    def _reset(self, data: mujoco.MjData):
        self._bad_locking = False
        self._update(data)  # Delegate to _update
    
    def _update(self, data):
        lid_qpos = data.qpos[self.lid_qposadr]
        if lid_qpos < self.lid_qpos_max - 0.01:
        # try to lock while lid is open
            self._bad_locking = True
        else:
            self._bad_locking = False
        if not self._bad_locking:
        # lock when lid is closed
            data.eq_active[self.lid_lock] = 1
            
class Centrifuge_tiangen_tgear_mini(System):
    
    def _reload(self, model: mujoco.MjModel):
        self.lid_lock = self.name2id(mujoco.mjtObj.mjOBJ_EQUALITY, 'lid-lock')
        self.lid_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'lid')
        self.lid_qpos_min = model.jnt_range[self.lid_joint, 0].item()
        self.lid_qposadr = model.jnt_qposadr[self.lid_joint].item()
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
        self.rotor_joint = self.name2id(mujoco.mjtObj.mjOBJ_JOINT, 'rotor')
        self.lid_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'lid')

class Thermal_mixer_eppendorf_c(System):
    def _configure(self):
        self.buttons = [
            ('speed', 'up', self.add_subsystem(FlatButton('button-speed-up'))),
            ('speed', 'down', self.add_subsystem(FlatButton('button-speed-down'))),
            ('time', 'up', self.add_subsystem(FlatButton('button-time-up'))),
            ('time', 'down', self.add_subsystem(FlatButton('button-time-down'))),
            ('temp', 'up', self.add_subsystem(FlatButton('button-temp-up'))),
            ('temp', 'down', self.add_subsystem(FlatButton('button-temp-down'))),
        ]
    
    def _reload(self ,model: mujoco.MjModel):
        self.speed_down_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-speed-down')
        self.speed_up_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-speed-up')
        self.temp_down_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-temp-down')
        self.temp_up_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-temp-up')
        self.time_down_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-time-down')
        self.time_up_site = self.name2id(mujoco.mjtObj.mjOBJ_SITE, 'button-time-up')
        self.display = self.name2id(mujoco.mjtObj.mjOBJ_TEXTURE, 'body-display')

    def _reset(self, data):
        status_bar_state = StatusBar(
            program_number=None,
            program_name=None,
            time_mode="temp_control",
            key_lock=False,
            speaker=True,
            interval_mix=False,
            thermotop=None,
            device_status=None,
        )
        main_param_state = MainParameter(
            time=Time(seconds=60),
            time_pause=False,
            set_temperature=25.0,
            actual_temperature=25.0,
            frequency=900,
        )
        ui_state = NoProgramMain(
            status_bar=status_bar_state, main_parameter=main_param_state
        )
        self.ui_state = ui_state
        self.ui_state_trajectory = [deepcopy(self.ui_state)]

    def _update(self, data):
        for name, direction, button in self.buttons:
            if button.is_pressed and button.pressed_time > 0.25:
                button.reset_pressed_time()
                sign = 1 if direction == 'up' else -1
                match name:
                    case 'speed':
                        self.ui_state.main_parameter.frequency += sign * 50
                    case 'time':
                        time = self.ui_state.main_parameter.time
                        time.seconds += sign * time.step_size
                    case 'temp':
                        self.ui_state.main_parameter.set_temperature += sign * 1.0
                    case _:
                        pass
        if self.ui_state == self.ui_state_trajectory[-1]:
            ui_state = self.ui_state_trajectory[-1]
        else:
            ui_state = deepcopy(self.ui_state)
        self.ui_state_trajectory.append(ui_state)

class UIStateCoordinator(System):
    def _configure(self):
        self._need_manager = True

    def _reload(self, model: mujoco.MjModel):
        self._thermal_mixers: list[Thermal_mixer_eppendorf_c] = self.manager.systems_by_type.get(Thermal_mixer_eppendorf_c, [])

    def _finish(self):
        ui_state = []
        for thermal_mixer in self._thermal_mixers:
            state = {
                "target": thermal_mixer.display,
                "trajectory": thermal_mixer.ui_state_trajectory,
            }
            ui_state.append(state)
        with open("ui_state.pkl", "wb") as f:
            import pickle
            pickle.dump(ui_state, f)

if __name__ == '__main__':
    from simulation import Manager

    mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

    # manager = Manager.from_file('model/instrument/vortex_mixer_genie_2.xml', [VortexMixerGenie2()])
    # manager.reload()
    # manager.run_passive(fps=60, realtime_factor=1)

    manager = Manager.from_file('model/instrument/thermal_cycler_biorad_c1000.xml', [ThermalCyclerBioradC1000()])
    manager.reload()
    manager.run_passive(fps=60, realtime_factor=1)

    # manager = Manager.from_file('model/scene/gallery.xml', [VortexMixerGenie2('/vortex_mixer_genie_2:')])
    # manager.reload()
    # manager.run_passive(fps=20, realtime_factor=1)
