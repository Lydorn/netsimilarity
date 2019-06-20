# Nabla Sim MNIST

### Setup environment

Dependencies can be found in ```requirements.txt```:
- matplotlib
- numpy
- PIL
- torch
- torchvision

However the [docker/pytorch-netsimilarity](docker/pytorch-netsimilarity) Docker image has everything installed so it can be easier to use it. 
Launch that Docker image with ```sh run-docker.sh```.

Execute```python main.py``` to launch the training.

Then open and run [notebooks/mnist_viz.ipynb](notebooks/mnist_viz.ipynb) to visualize the results. You can start a Jupyter notebook from the pytorch-netsimilarity Docker image with ```sh /start_jupyter.sh```.