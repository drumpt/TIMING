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
    explainer_list="ig deeplift gradientshap fo afo dynamask fit winit winitset dynamaskset"
    modeltype_list="gru mtand"

    top=10
    toppc=0.01

    for modeltype in ${modeltype_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --data mimic \
                --eval \
                --modeltype ${modeltype} \
                --explainer ${explainer} \
                --testbs 50 \
                --mask mam \
                --top ${top} \
                --toppc ${toppc} \
                --logfile mimic_${explainer}_${modeltype}_mam_${top} \
                --resultfile mimic_${explainer}_${modeltype}_mam_${top}.csv \
                2>&1 &
            i=$((i + 1))
        done
    done
}

test_all_masking() {
    modeltype_list="mtand seft"
    # modeltype_list="gru mtand"

    # explainer_list="gradientshap ig"
    # # explainer_list="ig fo afo fit dynamask winit"
    # # explainer_list="gradientshap_carryforward deeplift_carryforward ig_carryforward"
    # for modeltype in ${modeltype_list}; do
    #     for explainer in ${explainer_list}; do
    #         CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
    #             --data mimic \
    #             --eval \
    #             --modeltype ${modeltype} \
    #             --explainer ${explainer} \
    #             --testbs 25 \
    #             --logfile mimic_${explainer}_${modeltype}_all_masking \
    #             --resultfile mimic_${explainer}_${modeltype}_all_masking.csv \
    #             2>&1
    #         wait_n
    #         i=$((i + 1))
    #     done
    # done

    # explainer_list="deeplift fo afo fit dynamask winit winitsetzero winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
    explainer_list="deeplift fo afo fit dynamask winit winitsetzero winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
    # explainer_list="winit winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
    for modeltype in ${modeltype_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --data mimic \
                --eval \
                --modeltype ${modeltype} \
                --explainer ${explainer} \
                --testbs 50 \
                --logfile mimic_${explainer}_${modeltype}_all_masking \
                --resultfile mimic_${explainer}_${modeltype}_all_masking.csv \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done
}

verify_mtand() {
    modeltype_list="mtand"
    explainer_list="deeplift"
    for modeltype in ${modeltype_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --data mimic \
                --eval \
                --modeltype ${modeltype} \
                --explainer ${explainer} \
                --testbs 200 \
                --logfile mimic_${explainer}_${modeltype}_all_masking \
                --resultfile mimic_${explainer}_${modeltype}_all_masking.csv \
                2>&1
            wait_n
            i=$((i + 1))
        done
    done
}

wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=6

# test_corr_masking
# test_standard
# test_set
# test_seft
# test_mam
test_all_masking
# verify_mtand
