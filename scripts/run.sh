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

GPUS=(0 1 2 3 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=4
num_max_jobs=7

for cv in 0
do
    # o x x o o o o x o x
    # explainer_list="deep_lift gradient_shap lime dyna_mask extremal_mask gate_mask fit augmented_occlusion occlusion retain"

    explainer_list="gate_mask"
    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
            --model_type state \
            --explainers $explainer \
            --fold $cv \
            --testbs 30 \
            --areas 0.1 \
            --output-file state_cum_${cv}_results.csv \
            --device cuda:0 \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
done

# CUDA_VISIBLE_DEVICES=7 python mortality/main.py \
#         --model_type seft \
#         --explainers integrated_gradients \
#         --fold 0 \
#         --testbs 10 \
#         --device cuda:0 \



# fold 0 to 4
# model_type: state (gru), mtand, seft
# explainers: many..