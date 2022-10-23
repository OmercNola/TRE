#!/bin/bash

docker build -t "omer:v1" -f ./Dockerfile . && \
	        docker run --gpus 'all' --ipc=host \
	       	--rm -it -v `pwd`:`pwd` -w `pwd` omer:v1 $@
