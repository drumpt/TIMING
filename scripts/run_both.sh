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

GPUS=(3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=5

for cv in 0
do
    for top in 50
    do
        # o x x o o o o x o x
        # explainer_list="deep_lift gradient_shap lime dyna_mask extremal_mask gate_mask fit augmented_occlusion occlusion retain"

        #explainer_list="integrated_gradients_point_abs integrated_gradients_point"
        # explainer_list="integrated_two_stage_both gate_mask integrated_gradients_base_abs dyna_mask extremal_mask fit occlusion integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature integrated_gradients_max"
        
        # explainer_list="integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature"
        # explainer_list="our diff_abs integrated_gradients_base_abs integrated_gradients_online integrated_gradients_feature integrated_gradients_online_feature"
        explainer_list="extremal_mask_develop"

        for explainer in ${explainer_list}; do
            for lambda_3 in 10.0 1.0 0.1 0.01 0
            do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
                    --model_type state \
                    --explainers $explainer \
                    --fold $cv \
                    --testbs 30 \
                    --top $top \
                    --areas 0.1 \
                    --lambda-3 $lambda_3 \
                    --output-file state_cum_${cv}_${top}_results_ext_lr0.001_predict_keep_mask_both.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done

        # for explainer in ${explainer_list}; do
        #     CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
        #         --model_type state \
        #         --explainers $explainer \
        #         --fold $cv \
        #         --testbs 30 \
        #         --top $top \
        #         --skip_train_timex \
        #         --areas 0.1 \
        #         --lambda-1 0.1 \
        #         --output-file state_cum_${cv}_${top}_results_ext_lr0.001_predict.csv \
        #         --device cuda:0 \
        #         2>&1 &
        #     wait_n
        #     i=$((i + 1))
        # done

        # for explainer in ${explainer_list}; do
        #     CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
        #         --model_type state \
        #         --explainers $explainer \
        #         --fold $cv \
        #         --testbs 30 \
        #         --top $top \
        #         --skip_train_timex \
        #         --areas 0.1 \
        #         --lambda-1 0.0 \
        #         --lambda-2 0.0 \
        #         --output-file state_cum_${cv}_${top}_results_ext_lr0.001_predict.csv \
        #         --device cuda:0 \
        #         2>&1 &
        #     wait_n
        #     i=$((i + 1))
        # done

        # for explainer in ${explainer_list}; do
        #     CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python mortality/main.py \
        #         --model_type state \
        #         --explainers $explainer \
        #         --fold $cv \
        #         --testbs 30 \
        #         --top $top \
        #         --skip_train_timex \
        #         --areas 0.1 \
        #         --lambda-1 0.0 \
        #         --output-file state_cum_${cv}_${top}_results_ext_lr0.001_predict.csv \
        #         --device cuda:0 \
        #         2>&1 &
        #     wait_n
        #     i=$((i + 1))
        # done
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