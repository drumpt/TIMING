wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4 5 6)
NUM_GPUS=${#GPUS[@]}
i=2
num_max_jobs=7

# explainer_list="integrated_gradients_online integrated_gradients_feature integrated_gradients_base_abs dyna_mask occlusion"
# explainer_list="deep_lift"
# explainer_list="gradient_shap"
# explainer_list="fit WinIT"
# explainer_list="timex timex++"
explainer_list="timex"
for explainer in ${explainer_list}; do
    for cv in 0 1 2 3 4
    do
        for top in 100
        do
            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                --model_type state \
                --explainers $explainer \
                --data mimic3 \
                --fold $cv \
                --testbs 100 \
                --areas 0.2 \
                --top $top \
                --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
                --device cuda:0 \
                2>&1 &
            wait_n
            i=$((i + 1))
        done
    done
done

# explainer_list="gate_mask"
# for explainer in ${explainer_list}; do
#     for cv in 0 1 2 3 4
#     do
#         for top in 100
#         do
#             for mask_lr in 0.1
#             do
#                 for lambda_1 in 0.005
#                 do
#                     for lambda_2 in 0.01
#                     do
#                         CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
#                             --model_type state \
#                             --explainers $explainer \
#                             --data mimic3 \
#                             --fold $cv \
#                             --testbs 10 \
#                             --areas 0.2 \
#                             --lambda-1 $lambda_1 \
#                             --lambda-2 $lambda_2 \
#                             --mask_lr $mask_lr \
#                             --top $top \
#                             --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
#                             --device cuda:0 \
#                             2>&1 &
#                         wait_n
#                         i=$((i + 1))
#                     done
#                 done
#             done
#         done
#     done
# done

# explainer_list="extremal_mask"
# for explainer in ${explainer_list}; do
#     for cv in 0 1 2 3 4
#     do
#         for top in 100
#         do
#             for mask_lr in 0.01
#             do
#                 for lambda_1 in 0.01
#                 do
#                     for lambda_2 in 10
#                     do
#                         CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
#                             --model_type state \
#                             --explainers $explainer \
#                             --data mimic3 \
#                             --fold $cv \
#                             --testbs 10 \
#                             --areas 0.2 \
#                             --lambda-1 $lambda_1 \
#                             --lambda-2 $lambda_2 \
#                             --mask_lr $mask_lr \
#                             --top $top \
#                             --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
#                             --device cuda:0 \
#                             2>&1 &
#                         wait_n
#                         i=$((i + 1))
#                     done
#                 done
#             done
#         done
#     done
# done


# explainer_list="occlusion augmented_occlusion fa fit gradient_shap lime deep_lift"
# for explainer in ${explainer_list}; do
#     for cv in 0 1 2 3 4
#     do
#         for top in 100
#         do
#             CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
#                 --model_type state \
#                 --explainers $explainer \
#                 --data mimic3 \
#                 --fold $cv \
#                 --testbs 10 \
#                 --areas 0.2 \
#                 --top $top \
#                 --output-file state_mimic3_${cv}_${top}_results_baseline.csv \
#                 --device cuda:0 \
#                 2>&1 &
#             wait_n
#             i=$((i + 1))
#         done
#     done
# done