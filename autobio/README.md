# AutoBio

## Requirements
### System requirement
- Ubuntu >= 20.04 (24.04 recommanded)
- WSL supported

### Environment setup with conda
```bash
conda create -n autobio python=3.11
conda activate autobio
pip install 'mujoco==3.3.0' numpy scipy jax[cpu] toppra trimesh shapely triangle manifold3d sympy zstandard tqdm networkx usd-core ffmpeg imageio[ffmpeg] matplotlib scikit-image
```

## File structure
### folders
- assets/ *3D models for lab assets*
- grasp/ *forward kinematics, quaternion*
- logs/ *storing generated simulation data*
- model/ *simulation-ready lab assets/scene in MJCF language*
- packages/ *websocket for inference using remote server*

### files
#### task definition
- load_centrifuge_5430.py *load centrifuge 5430 rotor*
- mani_centrifuge_5430.py *close/open lid of centrifuge 5430*
- mani_centrifuge_5910.py *close/open lid of centrifuge 5910*
- mani_centrifuge_mini.py *close/open lid of cenrtrifuge mini (desktop)*
- mani_pipette.py *aspirate liquid with pipette*
- mani_thermal_cycler.py *close/open lid of thermal cycler*
- mani_thermal_mixer.py  *set parameters (time, temperature, frequency) on a mixer panel*
- mani_vortex_mixer.py *close/open lid of vortex mixer*
- pickup_centrifuge_tube *pick up a centrifuge tube from its rack*
- screw_loosen.py *unscrew centrifuge tube cap*
- screw_tighten.py *screw on centrifuge tube cap*
- transfer_centrifuge_tube.py *transfer a tube to a specified rack slot*

#### noteble utilities
- task.py *abstract class for AutoBio tasks*
- render.py *get visual output for each camera from offline simulation data*
- kinematics.py *provide inverse kinematics for ur5e and analytical IK for aloha arm*
- evaluate.py *for policy evaluation*
- instrument.py *define functionalities or behaviors for lab instruments*

## Data generation
Run python file of *task definition*
```bash
python ./[task_definition].py
```
Each simulation trajectory will be saved into `./logs/[task_name]/[timestamp]/`. All trajectory samples share a common `.mjb` file which stores the scene information in binary.

After simulation data acquired, you may run 
```bash
bash render.bash "[task_name]"
```
to get the visual output for each camera in the scenario.
