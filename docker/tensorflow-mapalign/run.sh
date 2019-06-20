#!/usr/bin/env bash

docker run --runtime=nvidia -it --rm -p 8888:8888 -v ~/netsimilarity:/workspace tensorflow-mapalign