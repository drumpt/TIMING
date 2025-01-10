### Train Model

# python mortality/main.py \
#     --model_type mtand \
#     --train True \
#     --explainers deep_lift \
#     --device cuda:4

python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deep_lift \
    --device cuda:4