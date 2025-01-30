wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2)
NUM_GPUS=${#GPUS[@]}
i=2
num_max_jobs=3

for cv in 0 1 2 3 4
do
    for top in 100
    do
        for prob in 0.3 0.5 0.7 
        do
            explainer_list="our_random"
            for explainer in ${explainer_list}; do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type state \
                    --explainers $explainer \
                    --data mimic3 \
                    --fold $cv \
                    --testbs 30 \
                    --areas 0.2 \
                    --top $top \
                    --prob $prob \
                    --output-file state_mimic3_${cv}_${top}_results_final.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done
    done
done
