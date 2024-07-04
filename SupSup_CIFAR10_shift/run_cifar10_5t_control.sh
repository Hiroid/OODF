#!/bin/bash

python experiments/GG/run.py --data='/data0/user/lxguo/Data' --dataset 'cifar10' --config 'cifar10_5t' --log_dir './logs/cifar10_5t' --name 'cifar10_5t' --round 2 --gpu-sets=5 --seeds 1 --save --original --amp --epochs 100

