This image requires the official pytorch image to be built on the system.
See the official PyTorch GitHub instructions to built it: https://github.com/pytorch/pytorch#docker-image

This Docker image has the following installed:
- CUDA, cuDNN, Miniconda, PyTorch (from the pytorch image)
- jsmin
- tqdm
- sklearn

Once the pytorch image has been built, you can build this image with:
```
sh build.sh
```

Or:
```
nvidia-docker build -t pytorch-netsimilarity --rm .
```

----------------------

Run container:
```
sh run.sh
```

Or (change the path to the netsimilarity folder if it is not in your home folder):
```
docker run --runtime=nvidia -it --rm -v ~/netsimilarity:/workspace pytorch-netsimilarity
```

Then you can launch Jupyter with ```sh /start_jupyter.sh```