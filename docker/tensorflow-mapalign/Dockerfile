FROM nvidia/cuda:9.0-devel-ubuntu16.04

# Install useful tools
RUN apt-get update && apt-get install -y \
    wget \
    git \
    sudo \
    libcudnn7=7.0.5.15-1+cuda9.0 \
    libcudnn7-dev=7.0.5.15-1+cuda9.0 \
    libgtk2.0 \
    fish

RUN rm -rf /var/lib/apt/lists/*

#--- Install Anaconda3 ---#
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh
RUN wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH
# Update conda:
RUN conda update conda

#--- Install Tensorflow ---#
RUN pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.1-cp36-cp36m-linux_x86_64.whl
RUN pip install update
# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

#--- Image geo-manipulation ----#
RUN conda install pyproj

# Install gdal
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt-add-repository ppa:ubuntugis/ubuntugis-unstable
RUN apt-get update
RUN apt-get install -y libgdal-dev
# See https://gist.github.com/cspanring/5680334:
RUN pip install gdal==2.2.0 --global-option=build_ext --global-option="-I/usr/include/gdal/"

# Install overpy:
RUN pip install overpy

# Install shapely:
RUN conda install -c conda-forge shapely -y

# JSON manipulation:
RUN conda install -c conda-forge jsmin -y

# Install tqdm for terminal progress bar:
RUN conda install -c conda-forge tqdm -y

# Cleanup
RUN apt-get clean && \
    apt-get autoremove

# Setup main volume
ENV WORKDIR=/workspace
VOLUME $WORKDIR
WORKDIR $WORKDIR

COPY start_jupyter.sh /

CMD fish
