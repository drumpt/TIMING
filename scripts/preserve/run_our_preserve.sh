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
num_max_jobs=5

for cv in 0 1 2 3 4
do
    for top in 1536
    do
        for num_segments in 50
        do
            for min_seg_len in 10
            do
                for max_seg_len in 48
                do
                    explainer_list="our"
                    for explainer in ${explainer_list}; do
                        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main_preserve.py \
                            --model_type state \
                            --explainers $explainer \
                            --data mimic3 \
                            --fold $cv \
                            --testbs 30 \
                            --areas 0.2 \
                            --top $top \
                            --num_segments $num_segments \
                            --min_seg_len $min_seg_len \
                            --max_seg_len $max_seg_len \
                            --output-file preserve_state_mimic3_${cv}_${top}_results.csv \
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
