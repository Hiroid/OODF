#!/bin/bash

# python experiments/GG/run.py --data='/data0/user/lxguo/Data' --dataset 'cifar10' --config 'cifar10_10t' --log_dir './logs/cifar10_10t' --name 'cifar10_10t' --round 0 --gpu-sets=0 --seeds 1 --save --original --amp --epochs 1

# python experiments/GG/run.py --data='/data0/user/lxguo/Data' --dataset 'cifar10' --config 'cifar10_5t' --log_dir './logs/cifar10_5t' --name 'cifar10_5t' --round 0 --gpu-sets=5 --seeds 1 --save --original --amp

python experiments/GG/run.py --data='/data0/user/lxguo/Data' --dataset 'cifar10' --config 'cifar10_5t' --log_dir './logs/cifar10_5t' --name 'cifar10_5t' --round 1 --gpu-sets=5 --seeds 1 --save --original --amp --batch-size 1024 --test-batch-size 1024 --epochs 100