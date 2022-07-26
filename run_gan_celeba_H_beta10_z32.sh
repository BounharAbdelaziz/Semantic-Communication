#! /bin/sh

python3 main.py --dataset celeba --seed 1 --lr 1e-4 --lambda_D 1 --lambda_G 1 --lambda_PCP 0 --norm_type bn2d  --beta1 0.5 --beta2 0.5 --objective H --model H --batch_size 64 --z_dim 32 --max_iter 1.5e6 --beta 10 --viz_name celeba_H_beta10_z32
