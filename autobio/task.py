from pathlib import Path

import numpy as np
import mujoco

from simulation import Manager
from serialize import MujocoSerializer

PROJECT_ROOT = Path(__file__).parent

MODEL_ROOT = PROJECT_ROOT / "model"
SCENE_ROOT = MODEL_ROOT / "scene"

LOG_ROOT = PROJECT_ROOT / "logs"

class Task:
    default_scene: Path
    default_task: str

    time_limit: float
    early_stop: bool

    Expert: type["Expert"]

    def __init__(self, manager: Manager):
        manager.reload()
        self.manager = manager
        self.spec = manager.spec
        self.model = manager.model
        self.data = manager.data
        self.dt = self.model.opt.timestep

        self.task = self.default_task
        self.task_info = {}

        self.serializer = None

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        return spec

    @classmethod
    def load(cls, scene: Path | None = None) -> mujoco.MjSpec:
        if scene is None:
            scene = cls.default_scene
        spec = mujoco.MjSpec.from_file(str(scene))
        return cls.prepare(spec)
    
    def set_serializer(self, serializer: MujocoSerializer | None = None, log_root: Path | None = None, log_name: str | None = None):
        """Set the serializer for the task."""
        if serializer is not None:
            self.serializer = serializer
        if log_root is None:
            log_root = LOG_ROOT / self.task
        self.serializer = MujocoSerializer(
            self.spec, self.model, self.data, self.task_info,
            log_root=log_root, log_name=log_name,
        )

    def reset(self, seed: int | None = None):
        """Reset the task to its initial state."""
        if seed is not None:
            np.random.seed(seed)
        return self.task_info

    def step(self):
        self.manager.step()
    
    def step_and_log(self, info: dict):
        self.manager.step()
        if self.serializer:
            self.serializer.record(info)

    def finish(self):
        if self.serializer:
            self.serializer.finish()
            with self.serializer.within_save_dir():
                self.manager.finish()
        self.serializer = None

    def check(self):
        """Check if the task is done and successful."""
        return False

class Expert(Task):
    def execute(self):
        """Execute the task with expert control."""
        pass

def get_task_class(name: str) -> type[Task]:
    """Get the task class by name."""
    if name == "pickup":
        from pickup import Pickup
        return Pickup
    elif name == "thermal_cycler_close" or name == "thermal_cycler_open":
        from mani_thermal_cycler import ThermalCyclerManipulate
        return ThermalCyclerManipulate
    elif name == "insert":
        from insert import Insert
        return Insert
    elif name == "pipette":
        from mani_pipette import Pipette
        return Pipette
    elif name == "screw_loose":
        from screw_loose_topp import ScrewLoose
        return ScrewLoose
    elif name == "screw_tighten":
        from screw_tighten_topp import ScrewTighten
        return ScrewTighten
    elif name == "insert_centrifuge_5430":
        from insert_centrifuge_5430 import InsertCentrifuge5430
        return InsertCentrifuge5430
    elif name == "thermal_mixer":
        from mani_thermal_mixer import ThermalMixerManipulate
        return ThermalMixerManipulate
    elif name == "centrifuge_5430_close_lid":
        from mani_centrifuge_5430 import Centrifuge5430Manipulate
        return Centrifuge5430Manipulate
    elif name == "centrifuge_5910_lid_close":
        from mani_centrifuge_5910 import Centrifuge5910Manipulate
        return Centrifuge5910Manipulate
    elif name == "centrifuge_mini_close_lid":
        from mani_centrifuge_mini import CentrifugeMiniManipulate
        return CentrifugeMiniManipulate
    elif name == "vortex_mixer":
        from vortex_mixer import VortexMixerManipulate
        return VortexMixerManipulate
    else:
        raise ValueError(f"Unknown task name: {name}")

def create_task(name: str, **kwargs):
    """Create a task instance by name."""
    task_class = get_task_class(name)
    spec = task_class.load()
    task = task_class(spec, **kwargs)
    task.task = name
    return task

def create_expert(name: str, **kwargs):
    """Create an expert instance by name."""
    task_class = get_task_class(name)
    expert_class = task_class.Expert
    spec = task_class.load()
    expert = expert_class(spec, **kwargs)
    expert.task = name
    return expert
