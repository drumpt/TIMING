test_all_masking() {
    cv="0"
    explainerseed=2345

    modeltype_list="mtand seft"
    # modeltype_list="gru"

    explainer_list="gradientshap ig"
    # explainer_list="ig fo afo fit dynamask winit"
    # explainer_list="gradientshap_carryforward deeplift_carryforward ig_carryforward"
    for modeltype in ${modeltype_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --eval \
                --data mimic \
                --cv ${cv} \
                --explainerseed ${explainerseed} \
                --modeltype ${modeltype} \
                --ckptpath ckpt/ckpt_${modeltype}_reversed_new \
                --batchsize 128 \
                --explainer ${explainer} \
                --testbs 25 \
                --logfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed}_reversed_new \
                --resultfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed}_reversed_new.csv \
                2>&1

            wait_n
            i=$((i + 1))
        done
    done

    # explainer_list="winit winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
    explainer_list="deeplift fo afo fit dynamask winit winitsetzero winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
    for modeltype in ${modeltype_list}; do
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                --eval \
                --data mimic \
                --cv ${cv} \
                --explainerseed ${explainerseed} \
                --modeltype ${modeltype} \
                --ckptpath ckpt/ckpt_${modeltype}_reversed_new \
                --batchsize 128 \
                --explainer ${explainer} \
                --testbs 50 \
                --logfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed}_reversed_new \
                --resultfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed}_reversed_new.csv \
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

GPUS=(0 1 2 3 4 5)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=6

test_all_masking
