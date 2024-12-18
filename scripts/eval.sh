test_corr_masking() {
    top_p_list="0 5 10 30 50 100"
    for top_p in ${top_p_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --eval \
            --explainer ig \
            --mask ${mask} \
            --top_p ${top_p} \
            --logfile mimic_${top_p}_m_inf \
            2>&1
        i=$((i + 1))
    done
}

test_mam() {
    mask=mam
    explainer_list="ig deeplift gradientshap fo afo fit winit dynamask"
    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --eval \
            --explainer ${explainer} \
            --mask ${mask} \
            --testbs 100 \
            --top 30 \
            --logfile mimic_${explainer}_${mask} \
            2>&1 &
        i=$((i + 1))
    done
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0

# test_corr_masking
test_mam
