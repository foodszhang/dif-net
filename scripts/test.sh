gpu=0

CUDA_VISIBLE_DEVICES=$gpu python code/evaluate.py \
    --name dif-net \
    --epoch 2000 \
    --dst_list knee_cbct \
    --split test \
    --num_views 10
