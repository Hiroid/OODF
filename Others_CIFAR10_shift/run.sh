#!/bin/bash
# python general_main.py --data cifar10 --cl_type nc --agent ICARL --epoch 25 --batch 128 --fix_order True --learning_rate 0.1 --weight_decay 0.00001 --shift
# python general_main.py --data cifar10 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000 --epoch 25 --fix_order True --shift
# python general_main.py --data cifar10 --cl_type nc --agent GDUMB --mem_size 5000 --mem_epoch 30 --minlr 0.0005 --clip 10 --epoch 25 --fix_order True --shift
# python general_main.py --data cifar10 --cl_type nc --agent CNDPM --epoch 25 --batch 128 --fix_order True --learning_rate 0.006 --weight_decay 0.00001 --shift

# python general_main.py --data cifar100 --cl_type nc --agent ICARL --retrieve random --update random --mem_size 5000 --fix_order True --num_tasks 100 --epoch 25 --shift
# python general_main.py --data cifar100 --cl_type nc --agent ER --retrieve random --update random --mem_size 5000 --fix_order True --num_tasks 100 --epoch 25 --shift
# python general_main.py --data cifar100 --cl_type nc --agent GDUMB --mem_size 1000 --mem_epoch 30 --minlr 0.0005 --clip 10 --fix_order True --num_tasks 100 --epoch 25 --shift
# python general_main.py --data cifar100 --cl_type nc --agent CNDPM --stm_capacity 1000 --classifier_chill 0.01 --log_alpha -300 --fix_order True --num_tasks 100 --epoch 25 --shift

python general_main.py --data cifar100 --cl_type nc --agent ICARL --retrieve random --update random --mem_size 5000 --fix_order True --num_tasks 100 --epoch 25 --shift --shift_idx_list 49 --shift_perc 0.5 --shift_eps 255 --shift_position 28 28 1 1 --num_runs 5 2>&1 | tee "./log/$(date +'%Y%m%d%H%M%S').log" 
