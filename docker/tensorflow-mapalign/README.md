Docker container with the following installed:
- CUDA, cuDNN
- Anaconda3
- Tensorflow 1.10.1
- OpenCV 3.2
- skimage
- pyproj
- gdal
- overpy

Build image:
```
sh build.sh
```

Or:
```
nvidia-docker build -t tensorflow-mapalign --rm .
```

----------------------

Run container:
```
sh run.sh
```

Or (change the path to the netsimilarity folder if it is not in your home folder):
```
docker run --runtime=nvidia -it --rm -v ~/netsimilarity:/workspace lydorn/tensorflow-mapalign
```


Then you can launch Jupyter with ```sh /start_jupyter.sh```