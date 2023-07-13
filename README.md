# Running VIMA model in the real world

Dependencies:
- clone `utils` [repo](https://github.com/andrey1908/utils) @70c0297 to `rozumarm_vima/utils`
- clone `camera_utils` [repo](https://github.com/andrey1908/camera_utils) @5b932aa to `rozumarm_vima/camera_utils`
- clone `rozumarm_vima_cv` [repo](https://github.com/andrey1908/rozumarm_vima_cv) @b200ac3 to `rozumarm_vima/rozumarm_vima_cv`
- clone `rozumarm_vima_utils` [repo](https://github.com/ag-cdsl/rozumarm-vima-utils) @64704fd and install as a package

```
export PYTHONPATH=${PYTHONPATH}:/home/daniil/code/rozumarm-vima/rozumarm_vima/rozumarm_vima_cv:/home/daniil/code/rozumarm-vima/rozumarm_vima/utils:/home/daniil/code/rozumarm-vima/rozumarm_vima/camera_utils
```

Download all missing VIMA checkpoints from https://github.com/vimalabs/VIMA 

How to:
- to start (aruco detector -> sim -> oracle -> arm) pipeline, run `scripts/run_aruco2sim_loop.py`
- to start (aruco detector -> sim -> VIMA model -> arm) pipeline, run `scripts/run_model_loop.py`