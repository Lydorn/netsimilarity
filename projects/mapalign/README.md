# Introduction

This folder contains all the scripts for the map alignment project.
This README presents the steps to analyse the pre-trained networks on the Inria Aerial Image Dataset with the developed similarity measures.

# Python environment

The code uses a few Python libraries such as Tensorflow, etc.
The docker image 
[tensorflow-mapalign](../../../docker/tensorflow-mapalign) has all the needed dependencies.
See the instructions in the [docker](../../../docker) folder to install docker and build that image.
Then start that Docker image with ```sh run-docker.sh```. 

# Download datasets

The dataset used for experiments is (see that folder for instructions on downloading):
- [Inria Aerial Image Dataset](../../../data/AerialImageDataset):

# Launch experiments

The useful scripts for experiments are in the [mapalign_multires](../mapalign_multires) folder.