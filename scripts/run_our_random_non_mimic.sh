wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=5

data=epilepsy

# for cv in 0 1 2 3 4
for cv in 0 1 2 3 4
do
    for top in 0
    do
        # for prob in 0.1 0.3 0.5 0.7 0.9
        # do
        # --prob $prob \
            explainer_list="integrated_gradients_base_abs"
            for explainer in ${explainer_list}; do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type state \
                    --explainers $explainer \
                    --data $data \
                    --fold $cv \
                    --testbs 30 \
                    --areas 0.1 \
                    --top $top \
                    --output-file state_${data}_${cv}_${top}_results_fix.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            #done
        done
    done
done
