# train_winit() {
#     CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
#         --data mimic \
#         --traingen \
#         --skipexplain
# }

# train_joint() {
#     CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
#         --data mimic \
#         --traingen \
#         --skipexplain \
#         --explainer fit
# }

train_forecastor() {
    forecastor_list="linear mlp"
    
    for forecastor in ${forecastor_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --traingen \
            --skipexplain \
            --forecastor ${forecastor} \
            --explainer deeplift_forecast \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
}

wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}


GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=4

# train_winit
# train_joint
train_forecastor