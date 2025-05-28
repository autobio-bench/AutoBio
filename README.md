# AutoBio [Preliminary Version]

⚠️ **Note**: This is currently a preliminary version of AutoBio, including our simulation assets and example code for synthetic data generation. The project is in development, and the codebase is undergoing structural improvements. We appreciate your understanding as we work to refine and stabilize the implementation. Contributions and suggestions are welcome! For details, please refer to our paper [AutoBio: A Simulation and Benchmark for Robotic Automation in Digital Biology Laboratory](https://arxiv.org/abs/2505.14030)

This codebase contains the following directories:

- `autobio`: The AutoBio codebase. Please refer to the `autobio/README.md` file for instructions on how to install and use AutoBio.

- `openpi`: The modified openpi ($\pi_0$) codebase adapted from [OpenPI](https://github.com/Physical-Intelligence/openpi) and contains our code to convert autobio data to LeRobot format, as well as the code to reproduce experiment results about $\pi_0$. Please refer to the `openpi/README.md` file for instructions on how to install and use the code.

- `RoboticsDiffusionTransformer`: The modified RDT codebase adapted from [RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer), which is used to reproduce experiment results about RDT. Our modifications are mainly about `RoboticsDiffusionTransformer/data/lerobot_vla_dataset.py`, which is used to load LeRobot datasets. Please refer to the `RoboticsDiffusionTransformer/README.md` file for instructions on how to install and use the code.

## Dataset
We host the datasets of all tasks in the paper on HuggingFace. It comes in two flavors: videos rendered by MuJoCo and videos rendered by Blender Cycles. The datasets are available at the following links:
- [MuJoCo dataset collection](https://huggingface.co/collections/autobio-bench/autobio-mujoco-68219f844a4650a970b307bd):
  - [Close thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_close-mujoco)
  - [Open thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_open-mujoco)
  - [Pick up centrifuge tube](https://huggingface.co/datasets/autobio-bench/pickup-mujoco)
  - [Unscrew centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_loose-mujoco)
  - [Aspirate with pipette](https://huggingface.co/datasets/autobio-bench/pipette-mujoco)
  - [Transfer centrifuge tube](https://huggingface.co/datasets/autobio-bench/insert-mujoco)
  - [Screw on centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_tighten-mujoco)
  - [Operate thermal mixer panel](https://huggingface.co/datasets/autobio-bench/thermal_mixer-mujoco)
  - [Load centrifuge rotor](https://huggingface.co/datasets/autobio-bench/insert_centrifuge_5430-mujoco)
- [Blender Cycles dataset collection](https://huggingface.co/collections/autobio-bench/autobio-blender-6824b2fbd77b18fe7a00595d)
  - [Close thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_close-blender)
  - [Open thermal cycler lid](https://huggingface.co/datasets/autobio-bench/thermal_cycler_open-blender)
  - [Pick up centrifuge tube](https://huggingface.co/datasets/autobio-bench/pickup-blender)
  - [Unscrew centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_loose-blender)
  - [Aspirate with pipette](https://huggingface.co/datasets/autobio-bench/pipette-blender)
  - [Transfer centrifuge tube](https://huggingface.co/datasets/autobio-bench/insert-blender)
  - [Screw on centrifuge tube cap](https://huggingface.co/datasets/autobio-bench/screw_tighten-blender)
  - [Operate thermal mixer panel](https://huggingface.co/datasets/autobio-bench/thermal_mixer-blender)
  - [Load centrifuge rotor](https://huggingface.co/datasets/autobio-bench/insert_centrifuge_5430-blender)

### Dataset format
All datasets follow the [LeRobot v2.0](https://github.com/huggingface/lerobot) format:
```
.
├── data
│   └── chunk-000
│       ├── episode_000000.parquet: Episode 0's data (numerical, images exluded), each row is a frame
│       └── ...
├── meta
│   ├── episodes.jsonl: Episode metadata
│   ├── info.json: Dataset metadata
│   ├── stats.json: Dataset statistics
│   └── tasks.jsonl: Task metadata
└── videos
    └── chunk-000
        ├── image: World-fixed camera
        │   ├── episode_000000.mp4: Episode 0
        │   └── ...
        └── ...: Other cameras
```
To quickly navigate the dataset, you can use the HuggingFace dataset viewer to check numerical data, and download specific videos from the dataset's `videos` directory to play. The videos are rendered at 224x224@50fps, which match OpenPi setting and save storage space. To check a high-resolution version, you can navigate to the `videos` directory of our supplementary material.

## Overall workflow

### Environment setup
Follow the instructions in the `autobio/README.md` file to set up the environment for AutoBio.

NOTE: Our codebase currently assumes Linux only. We have tested it on Ubuntu 20.04 and 24.04 (recommended).

NOTE: The dependencies of `openpi` and `RoboticsDiffusionTransformer` are somewhat fragile and shall be installed to separate environments. Evaluation of finetuned models can be achieved by remote inference.

### Data collection (Optional)
*If you prefer using our rendered videos, you can skip this step and download the datasets from HuggingFace.*

All commands below are run from the `autobio` directory and in the conda environment `autobio` created in the previous step.
1. **Trajectory collection**: Basic data collection is done by executing a specific task file directly. The synthesized trajectories are saved in the `autobio/logs/<task_name>` directory.
2. **Video rendering**: For MuJoCo rendering, run `bash render.bash logs/<task_name>` to render the videos for all trajectories in `autobio/logs/<task_name>`. You can adjust flags as needed. For Blender rendering, ensure that you have the Blender (>=4.4) installed and run `blender --background --python render_blender.py -- logs/<task_name>/<traj_name>/` to generate the Blend file at `logs/<task_name>/<traj_name>/scene.blend`. The Blender rendering then follows Blender's own workflow. An example command is

```bash
blender -b 'logs/<task_name>/<traj_name>/scene.blend' --render-output '<out_dir>/####.png' --engine CYCLES --render-anim --render-format PNG
ffmpeg -framerate 50 -i '<out_dir>/%04d.png' -filter:v 'format=rgba,premultiply=inplace=1' -c:v libx264 -pix_fmt yuv420p '<traj_name>.mp4'
```

### Data conversion to LeRobot format
*If you prefer using our rendered videos, you can skip this step and download the datasets from HuggingFace.*

All commands below are run from the `openpi` directory. Run:
```bash
LEROBOT_HOME=$PWD uv run scripts/convert.py --data_dir '../autobio/logs/<task_name>' --repo_id 'data/<task_name>'
LEROBOT_HOME=$PWD JAX_PLATFORMS=cpu uv run python scripts/compute_norm_stats.py --config-name '<task_name>'
```
This will convert the data to LeRobot format and save it in the `openpi/data/<task_name>` directory. The `compute_norm_stats.py` script computes normalization statistics for the data required by openpi.

### Training
See `openpi/slurm/train.bash` and `RoboticsDiffusionTransformer/train.bash` for slurm launchers for training scripts that are most close to our settings. Full training at batch size of 32 requires 1 GPU of 80GiB memory (such as NVIDIA A100, NVIDIA H100, or NVIDIA H800). For smaller GPUs (such as NVIDIA RTX 4090), openpi supports lora training, launching by config name `<task_name>-lora`.

### Evaluation
Evaluation of finetuned models can be achieved by remote inference. The first step is to run the policy server:
```bash
# for openpi (in openpi directory)
XLA_PYTHON_CLIENT_MEM_FRACTION=.6 CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint --policy.config '<task_name>' --policy.dir 'checkpoints/<task_name>/<exp_name>/29999'
# for RoboticsDiffusionTransformer (in RoboticsDiffusionTransformer directory and with the rdt environment activated)
python scripts/autobio_serve.py 'checkpoints/<exp_name>'
```
Then, you can run `autobio/evaluate.py` to evaluate the model.
```bash
python evaluate.py --port 8000 --task '<task_name>' --num_episodes 100 --image_history 0 --num_workers 0 --render_device_id 0 --save result.json
```
This will run the evaluation and save the results in `result.json`. You can adjust the parameters as needed.
