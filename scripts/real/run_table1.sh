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
            for num_segments in 50
            do
                for min_seg_len in 10
                do
                    for max_seg_len in 48
                    do
                        explainer_list="our"
                        for explainer in ${explainer_list}; do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                                --model_type $model \
                                --explainers $explainer \
                                --data mimic3 \
                                --fold $cv \
                                --testbs 30 \
                                --areas 0.1 \
                                --top $top \
                                --num_segments $num_segments \
                                --min_seg_len $min_seg_len \
                                --max_seg_len $max_seg_len \
                                --output-file ${model}_mimic3_${cv}_${top}_results.csv \
                                --device cuda:0 \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done

    explainer_list="timex++"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 100
            do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type $model \
                    --explainers $explainer \
                    --data mimic3 \
                    --fold $cv \
                    --testbs 50 \
                    --areas 0.1 \
                    --top $top \
                    --output-file ${model}_mimic3_${cv}_${top}_results_baseline.csv \
                    --device cuda:0 \
                    2>&1 &
                wait_n
                i=$((i + 1))
            done
        done
    done

    explainer_list="gate_mask"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 100
            do
                for mask_lr in 0.1
                do
                    for lambda_1 in 0.005
                    do
                        for lambda_2 in 0.01
                        do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                                --model_type $model \
                                --explainers $explainer \
                                --data mimic3 \
                                --fold $cv \
                                --testbs 10 \
                                --areas 0.1 \
                                --lambda-1 $lambda_1 \
                                --lambda-2 $lambda_2 \
                                --mask_lr $mask_lr \
                                --top $top \
                                --output-file ${model}_mimic3_${cv}_${top}_results_baseline.csv \
                                --device cuda:0 \
                                --deterministic \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done

    explainer_list="extremal_mask"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 100
            do
                for mask_lr in 0.01
                do
                    for lambda_1 in 0.01
                    do
                        for lambda_2 in 10
                        do
                            CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                                --model_type $model \
                                --explainers $explainer \
                                --data mimic3 \
                                --fold $cv \
                                --testbs 10 \
                                --areas 0.1 \
                                --lambda-1 $lambda_1 \
                                --lambda-2 $lambda_2 \
                                --mask_lr $mask_lr \
                                --top $top \
                                --output-file ${model}_mimic3_${cv}_${top}_results_baseline.csv \
                                --device cuda:0 \
                                --deterministic \
                                2>&1 &
                            wait_n
                            i=$((i + 1))
                        done
                    done
                done
            done
        done
    done

    explainer_list="integrated_gradients_base_abs integrated_gradients_base"
    for explainer in ${explainer_list}; do
        for cv in 0 1 2 3 4
        do
            for top in 100
            do
                CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                    --model_type $model \
                    --explainers $explainer \
                    --data mimic3 \
                    --fold $cv \
                    --testbs 10 \
                    --areas 0.1 \
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