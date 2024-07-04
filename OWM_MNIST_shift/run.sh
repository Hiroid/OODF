#!/bin/bash

# Control
python main.py 

# # Shift
# python main.py --shift

# # Defence
# python main.py --shift -n_e 30 --num_classes 11 -l_n_b True 2>&1 | tee "./log/$(date +'%Y%m%d%H%M%S').log"