#!/bin/bash

# iCaRL
python main.py --experiment splitMNIST --scenario class --tasks 10 --icarl --budget=2000

# # DGR
# python main.py --experiment splitMNIST --scenario class --tasks 10 --replay=generative --distill

# ER
# python main.py --experiment splitMNIST --scenario class --tasks 10 --replay=exemplars --budget=2000