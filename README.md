# Overview

Official Code for "Out-of-distribution forgetting: vulnerability of continual learning to intra-class distribution shift" [[PDF](https://arxiv.org/abs/2306.00427)]

In this repository, we provide the codes for splitMNIST-10, splitCIFAR-10&100. ([Link](https://github.com/Hiroid/OODF))

# Environments

See ```requirements.txt``` or ```README.md``` for the detailed environment in the folder of each method.



# Experiments
- If an experiment was produced in the paper, it is marked with ✅. Otherwise, it is marked with ❌.
- See ```run.sh``` for running command in the folder of each method, as well as the shift condition.

|          | OWM |  AOP | iCaRL | DGR | ER | GDumb | DER++ | CN-DPM | SupSup | 
| ---------| ------| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| MNIST | ✅ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ |
| CIFAR10 | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| CIFAR100 | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |