#!/bin/bash

python ./utils/main.py --model derpp  --alpha 0.1 --beta 1.0 --lr 0.03 --dataset seq-cifar10 --buffer_size 5120 --shift
