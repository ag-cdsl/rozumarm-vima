# Running VIMA model in the real world

Dependencies:
- clone `utils` [repo](https://github.com/andrey1908/utils) @compat_with_rozumarm_vima_cv_2023.05.26 to `rozumarm_vima/utils`
- clone `camera_utils` [repo](https://github.com/andrey1908/camera_utils) @compat_with_rozumarm_vima_cv_2023.05.26 to `rozumarm_vima/camera_utils`
- clone `rozumarm_vima_cv` [repo](https://github.com/andrey1908/rozumarm_vima_cv) @2023.05.26 to `rozumarm_vima/rozumarm_vima_cv`
- clone `rozumarm_vima_utils` [repo](https://github.com/ag-cdsl/rozumarm-vima-utils) @64704fd and install as a package

```
export PYTHONPATH=${PYTHONPATH}:/home/daniil/code/rozumarm-vima/rozumarm_vima/rozumarm_vima_cv:/home/daniil/code/rozumarm-vima/rozumarm_vima/utils:/home/daniil/code/rozumarm-vima/rozumarm_vima/camera_utils
```

How to:
- to start (segmentation -> model -> arm) pipeline, run `scripts/run_model_loop.py`