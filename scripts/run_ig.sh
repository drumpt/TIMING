# python mortality/main.py \
#     --model_type state \
#     --explainers integrated_gradients \
#     --fold 1 \
#     --device cuda:0

# python mortality/main.py \
#     --model_type state \
#     --explainers integrated_gradients_base \
#     --fold 1 \
#     --device cuda:0

# python mortality/main.py \
#     --model_type state \
#     --explainers integrated_gradients_base_cf \
#     --fold 1 \
#     --device cuda:0

# python mortality/main.py \
#     --model_type state \
#     --explainers integrated_gradients integrated_gradients_fixed integrated_gradients_base integrated_gradients_base_cf \
#     --fold 0 \
#     --device cuda:0 \

wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=6
num_max_jobs=5

for cv in 0
do
    for top in 50 100
    do
        # explainer_list="fa integrated_gradients_online integrated_gradients_base_abs integrated_gradients_feature"
        # explainer_list="gradientshap_abs gradientshap_online gradientshap_feature"
        # explainer_list="our_time"
        explainer_list="our our_v2"
        for explainer in ${explainer_list}; do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                --model_type state \
                --explainers $explainer \
                --data mimic3 \
                --fold $cv \
                --testbs 10 \
                --areas 0.1 \
                --top $top \
                --output-file state_mimic_cum_${cv}_${top}_results_method_fix.csv \
                --device cuda:0 \
                2>&1 &
            wait_n
            i=$((i + 1))
        done
    done
done
