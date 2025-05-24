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

model_list="state"
for model in ${model_list}; do
    for cv in 0 1 2 3 4
    do
        for top in 100
        do
            for prob in 0.3 0.5 0.7
            do
                explainer_list="our_random"
                for explainer in ${explainer_list}; do
                    CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                        --model_type $model \
                        --explainers $explainer \
                        --data mimic3 \
                        --fold $cv \
                        --testbs 30 \
                        --areas 0.2 \
                        --prob $prob \
                        --top $top \
                        --output-file ${model}_mimic3_${cv}_${top}_results_baseline.csv \
                        --device cuda:0 \
                        2>&1 &
                    wait_n
                    i=$((i + 1))
                done
            done
        done
    done
done