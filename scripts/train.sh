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

python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deeplift \
    --fold 0 \
    --device cuda:0 \

    
python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deeplift \
    --fold 1 \
    --device cuda:0 \

    
python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deeplift \
    --fold 2 \
    --device cuda:0 \

    
python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deeplift \
    --fold 3 \
    --device cuda:0 \

    
python mortality/main.py \
    --model_type seft \
    --train True \
    --explainers deeplift \
    --fold 4 \
    --device cuda:0 \