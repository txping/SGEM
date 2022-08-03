# SGEM
This repository contains code to reproduce the experiments in "SGEM: Stochastic gradient with energy and momentum".

## Usage
The sgdem.py file provides a PyTorch implementation of SGEM.

```python3
optimizer = sgdem.SGEM(model.parameters(), lr=0.2)
```

## Examples on CIFAR-10 and CIFAR-100
We compare SGEM with SGDM, Adam, AdaBelief, AdaBound, RAdam, Yogi and AEGD in six image classification tasks: VGG-16, ResNet-34, DenseNet-121 on CIFAR10 and CIFAT100. This repo heavily depends on the official implementation of [AEGD](https://github.com/txping/AEGD) and [AdaBelief](https://github.com/juntang-zhuang/Adabelief-Optimizer/tree/update_0.2.0/PyTorch_Experiments). We also provide a [notebook](./visualization.ipynb) to present our results for this example.

In this setting, the mini-batch size is set as `128` and the weight decay is set as `5e-4` for all tasks. We only tune the base learning rate and report the one that achieves the best final generalization performance for each method. The best base learning rate for each method in a certain task can be found in `curve/pretrained` fold to ease your reproduction.

Below is an example to train ResNet-34 on CIFAR-10 using SGEM with the default base learning rate 0.2.

```bash
python main.py --dataset cifar10 --model ResNet34 --optim SGEM --lr 0.2
```
The checkpoints will be saved in the `checkpoint` folder and the data points of the learning curve will be saved in the `curve` folder.


## License
[BSD-3-Clause](./LICENSE)
