# Net Similarity

This is the code from the submitted paper "Input Similarity from the Neural Network Perspective".

# Projects

### Toy problem

The code for the toy problem of the paper is in [projects/netsimilarity](projects/netsimilarity), see its README to run experiments and plot results.

### Map alignment problem

The code for the map alignment problem of the paper is in [projects/mapalign/mapalign_multires](projects/mapalign/mapalign_multires), see its README to run experiments and plot results.

### Nabla Sim MNIST

The code for the experiment about forcing similarities between pairs of training samples while training can be found in [projects/nabla_sim_mnist](projects/nabla_sim_mnist).

# Docker images

For easier environment setup, Docker images with everything installed inside are provided. For a brief introduction of Docker, see [docker](docker).

### Toy problem

The Docker image with PyTorch and other dependencies for the toy problem can be built with the instructions in the folder [docker/pytorch-netsimilarity](docker/pytorch-netsimilarity).

### Map alignment problem

The Docker image with Tensorflow and other dependencies for the map alignment problem can be built with the instructions in the folder [docker/tensorflow-mapalign](docker/tensorflow-mapalign).
