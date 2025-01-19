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
    # explainer_list="integrated_gradients integrated_gradients_point integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature integrated_gradients_base integrated_gradients_base_zero_cf gradient"
    # explainer_list="integrated_gradients_base integrated_gradients_base_zero_cf"
    # explainer_list="integrated_gradients_base_zero_cf"
    # explainer_list="gradient"
    # explainer_list="integrated_two_stage_both integrated_two_stage"
    # explainer_list="integrated_two_stage_both integrated_two_stage integrated_three_stage integrated_gradients integrated_gradients_point integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature integrated_gradients_base integrated_gradients_base_abs integrated_gradients_base_zero_cf"
    # explainer_list="integrated_two_stage_both integrated_two_stage integrated_three_stage integrated_gradients integrated_gradients_point integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature integrated_gradients_base integrated_gradients_base_abs integrated_gradients_base_zero_cf"
    # explainer_list="deeplift_two_stage_both deeplift_two_stage deeplift_three_stage deeplift deeplift_point deeplift_online deeplift_feature deeplift_online_feature deeplift_base deeplift_base_abs deeplift_base_zero_cf"

    # explainer_list="integrated_two_stage_both"
    explainer_list="integrated_gradients integrated_gradients_fixed integrated_gradients_point integrated_gradients_online integrated_gradients_feature integrated_gradients_feature_modify integrated_gradients_smooth diff integrated_two_stage integrated_two_stage_both integrated_random_mask integrated_three_stage integrated_gradients_online_feature integrated_gradients_max integrated_gradients_base integrated_gradients_base_abs integrated_gradients_base_cf integrated_gradients_base_zero_cf"

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