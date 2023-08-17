# Running VIMA model in the real world

Dependencies:
- clone & install [repo-1](https://github.com/andrey1908/kas_utils)
- clone & install [repo-2](https://github.com/andrey1908/kas_camera_utils)
- clone `rozumarm_vima_cv` [repo](https://github.com/andrey1908/rozumarm_vima_cv) to `rozumarm_vima/rozumarm_vima_cv`
- clone & install [repo-3](http://github.com/andrey1908/ultralytics)
- clone & install `rozumarm_vima_utils` [repo](https://github.com/ag-cdsl/rozumarm-vima-utils)

```
export PYTHONPATH=${PYTHONPATH}:/home/daniil/code/rozumarm-vima/rozumarm_vima/rozumarm_vima_cv:/home/daniil/code/rozumarm-vima/rozumarm_vima/utils:/home/daniil/code/rozumarm-vima/rozumarm_vima/camera_utils
```

Download all missing VIMA checkpoints from https://github.com/vimalabs/VIMA 

How to:
- to start (aruco detector -> sim -> oracle -> arm) pipeline, run `scripts/run_aruco2sim_loop.py`
- to start (aruco detector -> sim -> VIMA model -> arm) pipeline, run `scripts/run_model_loop.py`