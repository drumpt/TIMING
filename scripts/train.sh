### Train Model

# python mortality/main.py \
#     --model_type mtand \
#     --train True \
#     --explainers deep_lift \
#     --device cuda:4

# python mortality/main.py \
#     --model_type seft \
#     --train True \
#     --explainers deep_lift \
#     --device cuda:4

# python mortality/main.py \
#     --model_type state \
#     --train True \
#     --explainers deeplift \
#     --fold 0 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type state \
#     --train True \
#     --explainers deeplift \
#     --fold 1 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type state \
#     --train True \
#     --explainers deeplift \
#     --fold 2 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type state \
#     --train True \
#     --explainers deeplift \
#     --fold 3 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type state \
#     --train True \
#     --explainers deeplift \
#     --fold 4 \
#     --device cuda:0 \
wait_n() {
    background=($(jobs -p))
    echo ${num_max_jobs}
    if ((${#background[@]} >= num_max_jobs)); then
        wait -n
    fi
}

GPUS=(5 6 7)
NUM_GPUS=${#GPUS[@]}
i=0
num_max_jobs=3

# boiler epilepsy
for data in PAM
do
    for cv in 0 1 2 3 4
    do
        CUDA_VISIBLE_DEVICES=${GPUS[i % ${NUM_GPUS}]} python real/main.py \
            --model_type state \
            --train True \
            --data $data \
            --explainers empty \
            --fold $cv \
            --device cuda:0 \
            2>&1 &
        wait_n
        i=$((i + 1))
    done
done
# python mortality/main.py \
#     --model_type seft \
#     --train True \
#     --explainers deeplift \
#     --fold 1 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type seft \
#     --train True \
#     --explainers deeplift \
#     --fold 2 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type seft \
#     --train True \
#     --explainers deeplift \
#     --fold 3 \
#     --device cuda:0 \

    
# python mortality/main.py \
#     --model_type seft \
#     --train True \
#     --explainers deeplift \
#     --fold 4 \
#     --device cuda:0 \