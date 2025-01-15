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

GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=4
num_max_jobs=4

for cv in 0
do
    # explainer_list="integrated_gradients integrated_gradients_point integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature integrated_gradients_base integrated_gradients_base_zero_cf"
    # explainer_list="integrated_gradients_base integrated_gradients_base_zero_cf"
    # explainer_list="integrated_gradients_base_zero_cf"
    explainer_list="motif_ig"

    for explainer in ${explainer_list}; do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
            --model_type state \
            --explainers $explainer \
            --fold $cv \
            --testbs 100 \
            --device cuda:0
            # 2>&1 &
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