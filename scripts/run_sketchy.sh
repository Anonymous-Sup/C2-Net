cd /home/zhengwei/github/C2-Net/


# CUDA_VISIBLE_DEVICES=3 nohup python -u train.py --model C2_Net --dataset sketchy \
# --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 150 --val_epoch 1 --weight_decay 5e-4 --nesterov \
# --train_way 64 --test_way 64 --train_shot 1  --test_shot 1 \
# --alpha 0.5 --pre > train_sketchy.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u train.py --model C2_Net --dataset sketchy \
--gpu 1 --alpha 0.5 \
--opt sgd --lr 1e-1 --gamma 1e-1 --epoch 150 --val_epoch 10 --weight_decay 5e-4 --nesterov \
--train_way 5 --test_way 5 --train_shot 1  --test_shot 1 > train_sketchy.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python train.py --model C2_Net --dataset sketchy \
# --gpu 1 --alpha 0.5 \
# --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 150 --val_epoch 1 --weight_decay 5e-4 --nesterov \
# --train_way 5 --test_way 5 --train_shot 1  --test_shot 1 

# python3 train.py --model C2_Net --dataset stanford_dog --opt sgd --lr 1e-1 --gamma 1e-1 --epoch 150 --val_epoch 1 --weight_decay 5e-4 --nesterov --train_way 30 --train_shot 5 --train_transform_type 0 --test_shot 1 5 --alpha 0.5 --pre
