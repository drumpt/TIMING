# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
#     --data mimic \
#     --traingen \
#     --skipexplain

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
    --data mimic \
    --traingen \
    --skipexplain \
    --explainer fit
