#!/bin/bash

python experiments/GG/run.py --data='/data0/user/lxguo/Data' --dataset 'cifar10' --config 'cifar10_5t_shift' --log_dir './logs/cifar10_5t_shift' --name 'cifar10_5t_shift' --round 2 --gpu-sets=0 --seeds 1 --save --original --amp --epochs 100

