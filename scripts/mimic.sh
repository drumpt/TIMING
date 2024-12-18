# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
#     --data mimic \
#     --train \
#     --skipexplain

# CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
#     --data mimic \
#     --traingen \
#     --skipexplain

top_p_list="0 5 10 30 50 100"
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=1

for top_p in ${top_p_list}; do
    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
        --data mimic \
        --eval \
        --top_p ${top_p} \
        --logfile mimic_${top_p}_m_inf \
        2>&1
    i=$((i + 1))
done

CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
    --data mimic \
    --modeltype seft \
    --train \
    --skipexplain \
    --outpath output_debug \
    --outpath ckpt_debug \
    --plotpath plots_debug \
    --logpath logs_debug
