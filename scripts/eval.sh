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

test_standard() {
    # mask=mam
    explainer_list="winitset ig deeplift gradientshap fo afo dynamask fit winit"
    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --eval \
            --explainer ${explainer} \
            --logfile mimic_${explainer}_standard \
            2>&1 &
        i=$((i + 1))
    done
}

test_set() {
    # explainer_list="winitset ig deeplift gradientshap fo afo dynamask fit winit"
    explainer_list="dynamaskset"
    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --eval \
            --modeltype mtand \
            --ckptpath ckpt \
            --testbs 100 \
            --explainer ${explainer} \
            --logfile mimic_${explainer}_set \
            2>&1
        i=$((i + 1))
    done
}

test_seft() {
    modeltype=seft
    explainer_list="deeplift"
    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
            --data mimic \
            --eval \
            --modeltype ${modeltype} \
            --ckptpath ckpt \
            --testbs 2 \
            --explainer ${explainer} \
            --logfile mimic_${explainer}_${modeltype} \
            2>&1
        i=$((i + 1))
    done
}

test_mam() {
    mask=mam
    top_list="10 1 5 30"
    explainer_list="ig deeplift gradientshap fo afo dynamask fit winit winitset dynamaskset"
    for top in ${top_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --data mimic \
                --eval \
                --modeltype mtand \
                --explainer ${explainer} \
                --mask ${mask} \
                --testbs 50 \
                --top ${top} \
                --logfile mimic_${explainer}_${mask}_${top} \
                2>&1
            i=$((i + 1))
        done
    done
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=3

# test_corr_masking
# test_standard
# test_set
# test_seft
test_mam
