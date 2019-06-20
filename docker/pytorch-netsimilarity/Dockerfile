FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
    libgtk2.0 \
    fish

RUN conda install -c conda-forge jsmin -y

# Install tqdm for terminal progress bar
RUN conda install -c conda-forge tqdm -y

# Install sklearn
RUN conda install -c anaconda scikit-learn -y

RUN conda install -c anaconda jupyter

RUN conda install -c conda-forge matplotlib

# Cleanup
RUN apt-get clean && \
    apt-get autoremove

# Setup main volume
ENV WORKDIR=/workspace
VOLUME $WORKDIR
WORKDIR $WORKDIR

COPY start_jupyter.sh /

CMD fish
