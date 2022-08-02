# OptDARTS
This repository contains the code used in the **experiments of Optimizer Combination Analysis for Differentiable Neural Architecture Search** as a project of Spring 2022 CSED 490Y.

## Requirements
- python 3.10.4
- pytorch 1.11.0
- tqdm 4.64.0
- numpy 1.21.5
- pandas 1.4.2
- [NAS-Bench-201-benchmark-file](https://github.com/D-X-Y/NAS-Bench-201): please download the bench mark file.

## References
- [Beta DARTS](https://github.com/Sunshine-Ye/Beta-DARTS) : NAS-Bench-201 Related Code Reference
- [DARTS-](https://github.com/Meituan-AutoML/DARTS-) : Landscape visualization Related Code Reference

## CAUTION!
I used [wandb](https://wandb.ai/site) to trace all state of searching, but it uses private account, so I delete it from the codes.

## Usage
### Architecture Search
```
python train_model.py
```

### Landscape Visualization
```
python landscape.py --exp-name <After searching, your experiment directory>
```

### Options for Architecture Search
--data *(location of the data corpus)* : Path to be saved dataset

--bench-data *(location of the benchmark corpus)* : Path for downloaded nas bench 201 benchmark file.

--unrolled : If you add this, you use 2nd order gradient approximation.

--perturb_alpha random : Use perturbation for architecture parameters. If you add this, you use SDARTS algorithm for architecture search.

--auxiliary_skip : Use auxiliary skip connection for architecture search. If you add this, you use DARTS- algorithm for architecture search.

--optimizer : Choose your weight optimizer, **sgd** or **adam**.

--arch_optimier : Choose your architecture optimizer, **sgd** or **adam**.

--sgd_learning_rate : Set your initial learning rate for SGD weight optimizer.

--sgd_learning_rate_min : Set your minial learning rate for SGD weight optimizer.

--sgd_momentum : Set your momentum for SGD weight optimizer

--adam_learning_rate : Set your initial learning rate for Adam weight optimizer.

--adam_learning_rate_min : Set your minimal learning rate for Adam weight optimizer.

--adam_beta1 : Set your beta1 hyperparameter for Adam weight optimizer. (Actually, it is similar to momentum hyperparameters in SGD optimizer)

--arch_sgd_learning_rate : Set your learning rate for SGD architecture optimizer.

--arch_sgd_momentum : Set your momentum for SGD architecture optimizer.

--arch_adam_learning_rate : Set your learning rate for Adam architecture optimizer.

--arch_adam_beta1 : Set your beta1 hyperparameter for Adam architecture optimizer.
