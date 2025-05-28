## Raw mesh extraction via PGSR
1. Clone PGSR repository from Github
```bash
git clone git@github.com:zju3dv/PGSR.git
```
install the environment according to the project's guidance

2. Convert the sequence of images into 3DGS asset and extract mesh
```bash
python train.py -s "./asset/processed/$ASSET" -m "./gs/$ASSET" --max_abs_split_points 0 --opacity_cull_threshold 0.05
python render.py -m "./gs/$ASSET" --max_depth 10.0 --voxel_size 0.01 --skip_test
```