python mortality/main.py \
    --model_type state \
    --explainers integrated_gradients \
    --fold 1 \
    --device cuda:4

# fold 0 to 4
# model_type: state (gru), mtand, seft
# explainers: many..