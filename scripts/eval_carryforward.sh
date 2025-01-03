test_all_masking() {
    for cv in {0..4}; do
        explainerseed=2345
        modeltype_list="gru"
        # # explainer_list="ig_forecast"
        # explainer_list="gradientshap ig gradientshap_carryforward gradientshap_forecast ig_carryforward ig_forecast"
        # # explainer_list="ig fo afo fit dynamask winit"
        # # explainer_list="gradientshap_carryforward deeplift_carryforward ig_carryforward"
        # for modeltype in ${modeltype_list}; do
        #     for explainer in ${explainer_list}; do
        #         CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
        #             --eval \
        #             --data mimic \
        #             --cv ${cv} \
        #             --explainerseed ${explainerseed} \
        #             --modeltype ${modeltype} \
        #             --ckptpath ckpt \
        #             --batchsize 100 \
        #             --explainer ${explainer} \
        #             --testbs 25 \
        #             --logfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed} \
        #             --resultfile mimic_${explainer}_${modeltype}_all_masking_${cv}_${explainerseed}.csv \
        #             2>&1 &
        #         wait_n
        #         i=$((i + 1))
        #     done
        # done

        # explainer_list="winit winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
        # explainer_list="deeplift deeplift_carryforward deeplift_forecast fo afo fit dynamask winit winitsetzero winitsetzerolong winitsetcf fitsetzero fitsetcf fozero"
        # explainer_list="winithidden deeplift_forecast gradientshap_forecast ig_forecast"
        explainer_list="ig deeplift gradientshap ig_carryforward deeplift_carryforward gradientshap_carryforward "
        p_list="0.0 0.1 0.2 0.5 1.0"
        for modeltype in ${modeltype_list}; do
            for explainer in ${explainer_list}; do
                for p in ${p_list}; do
                    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m winit.run \
                        --eval \
                        --data mimic \
                        --cv ${cv} \
                        --explainerseed ${explainerseed} \
                        --modeltype ${modeltype} \
                        --ckptpath ckpt \
                        --batchsize 100 \
                        --explainer ${explainer} \
                        --p ${p} \
                        --testbs 25 \
                        --logfile mimic_${explainer}_pseudo_${p}_${modeltype}_all_masking_${cv}_${explainerseed} \
                        --resultfile mimic_${explainer}_pseudo_${p}_${modeltype}_all_masking_${cv}_${explainerseed}.csv \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                done
            done
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
