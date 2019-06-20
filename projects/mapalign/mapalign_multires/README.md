# Analyse pre-trained networks with netsimilarity measures

### 1 - Download pre-trained networks
12 pe-trained networks are available: 3 rounds of multiple-rounds training was performed and each one trains 4 networks with downsampling factors {8, 4, 2, 1}.\
You can either: Execute the download script [download_pretrained.py](download_pretrained.py)\
Or: manually download the zipfile at https://dl.dropbox.com/s/tojzm31gocbk13g/runs.igarss2019.zip?dl=1,
extract and place the runs.igarss2019 folder in the [mapalign_multires](../mapalign_multires) folder (this README's folder)
so that the folder structure is ```projects/mapalign/mapalign_multires/runs.igarss2019```.

### 2 - Execute script
Execute the ```main_netsimilarity_neighbors.py``` script in the Tensorflow Docker image (or you own environment) to launch netsimilarity measure computation.
See ```python main_netsimilarity_neighbors.py --help``` for an explanation of each argument. Also here are a few exemples of how to use this script:
- ```python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all   --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_0```
- ```python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all_1 --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_1```
- ```python main_netsimilarity_neighbors.py --run_name ds_fac_4_inria_bradbury_all_2 --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_2```
- ```python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_0```
- ```python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_1```
- ```python main_netsimilarity_neighbors.py --mode individual --ds_fac 4 --output_dirname netsimilarity_ds_fac_4_round_2```
Some more parameters of the scriptt are defined as global variables at the beginning of the file, if you wish to change things even more.

Then to plot the results, use the ```main_netsimilarity_neighbors_plot.py``` outside of Docker so that it can display the plots. This script does not have any dependencies other that matplotblib.
See ```python main_netsimilarity_neighbors.py --help``` for an explanation of each argument. Also here are a few exemples of how to use this script:
- ```python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_0```
- ```python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_1```
- ```python main_netsimilarity_neighbors_plot.py --output_dirname netsimilarity_ds_fac_4_round_2```
- ```python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_0```
- ```python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_1```
- ```python main_netsimilarity_neighbors_plot.py --mode individual --output_dirname netsimilarity_ds_fac_4_round_2```

# Run management

A simple run management system has been implemented to handle different runs efficiently.
It is based on identifying a run by its name and timestamp. The argument ```--run_name``` refers to the last run by this name based on its timestamp.
If a new training session is started with the ```--new_run``` flag and a previously used name, the two runs will be differentiated by their timestamp.
More info on the run management system can be found in the script [run_utils.py](../../utils/run_utils.py)

# Brief explanation of other scripts

All other scripts in this folder are not meant to be run on their own, they are used by the other scripts.

[loss_utils.py](loss_utils.py) implements loss functions built in the TF graph (used by model.py)

[model.py](model.py) implements the MapAlignModel Python class used to create the model, optimize it and run it. It also implements gradient computation for use in netsimilarity measures.

[model_utils.py](model_utils.py) implements the neural network building functions (used by model.py)

# Glossary

Here is a list of words used throughout the project with their corresponding definition as some can be ambiguous.

| Word | Definition |
| ------ | ---------- |
| tile    | square portion of an image produced by pre-processing so that very big images are split into manageable pieces     |
| patch   | square crop of a tile produced by online-processing which is fed to the network (crop needed to fit the network's input and output parameters)    |
| layer   | regular NN layer    |
| level   | resolution level within a model/network. It is a set of layers whose inputs are from the same spatial resolution     |
