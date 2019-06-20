# Toy problem

## Setup environment

### Use Docker

The quickest way to run the code is to use the provided PyTorch Docker image which has all dependencies already installed, see [/docker/pytorch-netsimilarity](/docker/pytorch-netsimilarity) for instructions to built that image.
Once it is built, run it with ```sh run-docker.sh```. It will run the image with the repo folder linked inside the container (the repo folder must be in your home folder for this to work, otherwise see more detailed instructions about Docker to link specific volumes).

### Install everything yourself

Required dependencies:
- CUDA (if using GPU)
- cuDNN (if using GPU)
- Miniconda3
- PyTorch 1.1
- jsmin
- tqdm
- sklearn

## Launch experiments

Experiments need the PyTorch environment above all setup, or the relevant Docker image.
To launch all the experiments for the toy problem, execute the script [main_multiple_exps.py](main_multiple_exps.py) with the default arguments like so:
```
python main_multiple_exps.py --new_exp
```
Execute ```python main_multiple_exps.py --help``` for a description of each argument.

The default setting lunch experiments for f (frequency) in {1, 2, 4, 8} and n (number of sample) equal 2048. Each experiment is run 5 times and the median value of their results is computed to avoid result variation due to the stochastic nature of network optimization.

## Plot results

One the experiments have finished running, it is possible to plot the saved results.
Plotting the results is done outside of the Docker image as it requires only matplotlib which needs a display to plot the results.

Plotting the neighbor count of every experiment at once can be done with:
```
python main_plot_exps.py -x alpha -y curvature --y_log -z neighbors_soft
```
Other measures can be plotting by changing the -z argument. Execute ```python main_plot_exps.py --help``` for a description of each argument as well.

Plotting the various ways (different measures) to compute the neighbor count can be done with:
```
python main_plot_all_measures.py
```
