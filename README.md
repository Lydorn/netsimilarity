# Net Similarity

This is the official code for the paper:

**Input Similarity from the Neural Network Perspective**\
[Guillaume Charpiat](https://www.lri.fr/~gcharpia/),
[Nicolas Girard](https://www-sop.inria.fr/members/Nicolas.Girard/),
Loris Felardos,
[Yuliya Tarabalka](https://www-sop.inria.fr/members/Yuliya.Tarabalka/)\
NeurIPS 2019\
**\[Final version of paper soon available\]**

# Projects

### Toy problem

The code for the toy problem of the paper is in [projects/netsimilarity](projects/netsimilarity), see its README to run experiments and plot results.

### Map alignment problem

The code for the map alignment problem of the paper is in [projects/mapalign/mapalign_multires](projects/mapalign/mapalign_multires), see its README to run experiments and plot results.

### Nabla Sim MNIST

The code for the experiment about forcing similarities between pairs of training samples while training can be found in [projects/nabla_sim_mnist](projects/nabla_sim_mnist).

# Analyze the input similarity of your own network

## Compute gradients

To analyze the input similarity of any network, the first step is to compute all gradients of some input. This is specific to the deep learning framework you use. We used Tensorflow for the Map alignment problem and PyTorch for the toy problem so you can see how it is done for both.
For Tensorflow, see [projects/mapalign/mapalign_multires/model.py](projects/mapalign/mapalign_multires/model.py) in the ```compute_grads()``` method of the MapAlignModel class. This one is a bit more complex as the outout is 2D (displacement vector) so gradients are computed twice: for x and for y.
For PyTorch see [projects/netsimilarity/model.py](projects/netsimilarity/model.py) in the ```compute_grads()``` method of the MapAlignModel class. Here the output is 1D so gradients are computed once but the code can be adapted to more dimensions by changing the values of the ```d``` variable inside the method. 

## Compute similarity measures

Then all functions used to compute similarity measures are in [projects/utils/netsimilarity_utils.py](projects/utils/netsimilarity_utils.py).
If the gradients take too much memory, we implemented functions which read gradients from disk and do not load all of them in memory for computation.

# Docker images

For easier environment setup, Docker images with everything installed inside are provided. For a brief introduction of Docker, see [docker](docker).

### Toy problem

The Docker image with PyTorch and other dependencies for the toy problem can be built with the instructions in the folder [docker/pytorch-netsimilarity](docker/pytorch-netsimilarity).

### Map alignment problem

The Docker image with Tensorflow and other dependencies for the map alignment problem can be built with the instructions in the folder [docker/tensorflow-mapalign](docker/tensorflow-mapalign).

### If you use this code for your own research, please cite:

```
@InProceedings{Charpiat_2019_NeurIPS,
author = {Charpiat, Guillaume and Girard, Nicolas and Felardos, Loris and Tarabalka, Yuliya},
title = {Input Similarity from the Neural Network Perspective},
booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
month = {December},
year = {2019}
}
```