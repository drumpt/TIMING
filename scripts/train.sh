train_gru() {
    CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
        --data mimic \
        --train \
        --skipexplain
}

train_set_functions() {
    # modeltype_list="mtand seft"
    modeltype_list="mtand"
    for modeltype in ${modeltype_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --modeltype ${modeltype} \
            --train \
            --lr 0.001 \
            --skipexplain \
            --outpath output/output_${modeltype}_reversed \
            --ckptpath ckpt/ckpt_${modeltype}_reversed \
            --plotpath plots/plots_${modeltype}_reversed \
            --logpath logs/logs_${modeltype}_reversed \
            2>&1
        i=$((i + 1))
    done
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=7

# train_gru
train_set_functions
