wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(0 1 2 3 4)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=5

data=freezer

# for cv in 0 1 2 3 4
for cv in 0 1 2 3 4
do
    for top in 0
    do
        for num_segments in 1 5 10
        do
            for min_seg_len in 1 10
            do
                for max_seg_len in 301 100 10
                do
                    explainer_list="our"
                    for explainer in ${explainer_list}; do
                        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
                            --model_type state \
                            --explainers $explainer \
                            --data $data \
                            --fold $cv \
                            --testbs 30 \
                            --areas 0.1 \
                            --top $top \
                            --num_segments $num_segments \
                            --min_seg_len $min_seg_len \
                            --max_seg_len $max_seg_len \
                            --output-file state_${data}_${cv}_${top}_results_ht.csv \
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
