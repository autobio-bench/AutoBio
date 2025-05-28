from typing import Iterable, TypeVar, Callable
import time
import math

import mujoco, mujoco.viewer

T = TypeVar('T')
Loader = Callable[[], tuple[mujoco.MjModel, mujoco.MjData, mujoco.MjSpec | None]]

class System:
    def __init__(self, local_prefix: str = '', local_suffix: str = '', **kwargs):
        self.model = None
        self.local_prefix = local_prefix
        self.local_suffix = local_suffix
        self.parent_prefix = ''
        self.parent_suffix = ''
        self.subsystems: list['System'] = []
        self._need_manager = False
        self.finalized = False
        self._configure(**kwargs)
        self.finalized = True
        if self._need_manager:
            self.manager: 'Manager' = None

    @property
    def prefix(self) -> str:
        return self.parent_prefix + self.local_prefix

    @property
    def suffix(self) -> str:
        return self.local_suffix + self.parent_suffix
    
    @property
    def flat(self) -> list['System']:
        """Returns a flat list of all subsystems and itself."""
        flat = [self]
        for subsystem in self.subsystems:
            flat.extend(subsystem.flat)
        return flat

    def add_subsystem(self, subsystem: 'T') -> T:
        assert not self.finalized, f'System {self.__class__.__name__} is already finalized'
        assert isinstance(subsystem, System), f'Subsystem must be an instance of System, got {type(subsystem)}'
        self.subsystems.append(subsystem)
        return subsystem

    def propagate_namespace(self, prefix: str = '', suffix: str = ''):
        self.parent_prefix = prefix
        self.parent_suffix = suffix
        for subsystem in self.subsystems:
            subsystem.propagate_namespace(self.prefix, self.suffix)

    def make_name(self, name: str) -> str:
        """Creates a name with the prefix and suffix."""
        return f'{self.prefix}{name}{self.suffix}'

    def name2id(self, type: mujoco.mjtObj, name: str = "") -> int:
        full_name = self.make_name(name)
        id = mujoco.mj_name2id(self.model, type, full_name)
        if id == -1:
            raise ValueError(f'{full_name} not found for {type}')
        return id

    def reload(self, model: mujoco.MjModel):
        """Called when the model is changed (or first loaded)."""
        self.model = model
        for subsystem in self.subsystems:
            subsystem.reload(model)
        self._reload(model)

    def reset(self, data: mujoco.MjData):
        """Called when the data is reset."""
        assert self.model is not None, f'Model not loaded for {self.__class__.__name__}'
        for subsystem in self.subsystems:
            subsystem.reset(data)
        self._reset(data)

    def update(self, data: mujoco.MjData):
        """Called when the data is updated."""
        assert self.model is not None
        for subsystem in self.subsystems:
            subsystem.update(data)
        self._update(data)
    
    def visualize(self, data: mujoco.MjData, scene: mujoco.MjvScene):
        """Called when the data is visualized."""
        assert self.model is not None
        for subsystem in self.subsystems:
            subsystem.visualize(data, scene)
        self._visualize(data, scene)
    
    def finish(self):
        """Called when one simulation is finished. Override this for custom cleanup logic."""
        for subsystem in self.subsystems:
            subsystem.finish()
        self._finish()

    def _configure(self, **kwargs):
        """Called when the system is created. Override this to configure the system."""

    def _reload(self, model: mujoco.MjModel):
        """Called when the model is changed (or first loaded). Override this for custom reload logic."""

    def _reset(self, data: mujoco.MjData):
        """Called when the data is reset. Override this for custom reset logic."""

    def _update(self, data: mujoco.MjData):
        """Called when the data is updated. Override this for custom update logic."""

    def _visualize(self, data: mujoco.MjData, scene: mujoco.MjvScene):
        """Called when the data is visualized. Override this for custom visualization logic."""
    
    def _finish(self):
        """Called when one simulation is finished. Override this for custom cleanup logic."""

class Manager:
    def __init__(self, loader: Loader, systems: Iterable[System]):
        self.model = None
        self.data = None
        self.spec = None
        self.loader = loader
        
        systems = tuple(systems)
        visited = set()
        def validate_system(system: System):
            if id(system) in visited:
                raise ValueError(f'Circular or duplicate reference to system {system.__class__.__name__}')
            visited.add(id(system))
            assert isinstance(system, System), f'System must be an instance of System, got {type(system)}'
            for subsystem in system.subsystems:
                validate_system(subsystem)
        for system in systems:
            validate_system(system)

        for system in systems:
            system.propagate_namespace()

        systems_by_type: dict[type[System], list[System]] = {}
        for system in systems:
            for subsystem in system.flat:
                subsystem.manager = self  # Set the manager reference for each subsystem
                for cls in subsystem.__class__.__mro__:
                    if not issubclass(cls, System):
                        continue
                    category = systems_by_type.setdefault(cls, [])
                    category.append(subsystem)
        self.systems = systems
        self.systems_by_type = systems_by_type

    @staticmethod
    def from_model(model: mujoco.MjModel, systems: Iterable[System]):
        def loader():
            return model, mujoco.MjData(model), None
        return Manager(loader, systems)
    
    @staticmethod
    def from_spec(spec: mujoco.MjSpec, systems: Iterable[System]):
        def loader():
            model = spec.compile()
            return model, mujoco.MjData(model), spec
        return Manager(loader, systems)

    @staticmethod
    def from_file(path: str, systems: Iterable[System]):
        def loader():
            spec = mujoco.MjSpec.from_file(path)
            model = spec.compile()
            return model, mujoco.MjData(model), spec
        return Manager(loader, systems)
    
    @staticmethod
    def from_binary(path: str, systems: Iterable[System]):
        def loader():
            model = mujoco.MjModel.from_binary_path(path)
            return model, mujoco.MjData(model), None
        return Manager(loader, systems)

    def reload(self):
        self.model, self.data, self.spec = self.loader()
        for system in self.systems:
            system.reload(self.model)

    def reset(self, keyframe: int | None = None):
        if keyframe is not None:
            mujoco.mj_resetDataKeyframe(self.model, self.data, keyframe)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        for system in self.systems:
            system.reset(self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_rnePostConstraint(self.model, self.data)
        for system in self.systems:
            system.update(self.data)

    def visualize(self, scene: mujoco.MjvScene):
        scene.ngeom = 0
        for system in self.systems:
            system.visualize(self.data, scene)

    def finish(self):
        for system in self.systems:
            system.finish()

    def run_passive(self, fps: float = 60.0, realtime_factor: float = 1.0):
        """
        Simple passive viewer for the model, with several keys configured:
        - R: reload the model
        - Space: pause/resume
        - /: enter debugger (ipdb)
        """
        assert fps > 0 and realtime_factor >= 0  # realtime_factor == 0 means as fast as possible
        def adjust_step(timestep: float):
            step_per_frame = realtime_factor / fps / timestep
            step_per_frame = int(math.ceil(step_per_frame))
            real_fps = realtime_factor / timestep / step_per_frame
            print(f'Running at {real_fps:.2f} FPS; Step {step_per_frame} times per frame')
            timespan = 1 / real_fps
            return step_per_frame, timespan
        if realtime_factor == 0:
            step_per_frame = 0
            timespan = 1 / fps
        else:
            step_per_frame, timespan = adjust_step(self.model.opt.timestep)
        keys = {}
        def key_callback(key):
            keys[key] = True
        def handle_key(key):
            return keys.pop(key, False)
        def sync(handle: mujoco.viewer.Handle):
            self.visualize(handle.user_scn)
            handle.sync()
        pause = False
        with mujoco.viewer.launch_passive(self.model, self.data, key_callback=key_callback) as handle:
            mujoco.mjv_defaultFreeCamera(self.model, handle.cam)
            sync(handle)
            last_time = time.perf_counter()
            while handle.is_running():
                if handle_key(ord('R')):
                    sim = handle._get_sim()
                    if sim is not None:
                        self.reload()
                        sim.load(self.model, self.data, "")
                if handle_key(ord(' ')):
                    pause = not pause
                if handle_key(ord('/')):
                    import ipdb; ipdb.set_trace()
                
                if pause:
                    mujoco.mjv_applyPerturbPose(self.model, self.data, handle.perturb, 1)
                    mujoco.mj_forward(self.model, self.data)
                    sync(handle)
                    time.sleep(0.1)
                    continue
                if self.data.time == 0:
                    self.reset()
                if realtime_factor == 0:
                    current_time = time.perf_counter()
                    self.step()
                    while current_time - last_time < timespan:
                        current_time = time.perf_counter()
                        self.step()
                    last_time = current_time
                else:
                    for _ in range(step_per_frame):
                        self.step()
                    current_time = time.perf_counter()
                    if current_time - last_time < timespan:
                        time.sleep(timespan - (current_time - last_time))
                    else:
                        print(f'WARN: Frame time exceeded by {current_time - last_time - timespan:.2f}s')
                    last_time = time.perf_counter()
                sync(handle)
