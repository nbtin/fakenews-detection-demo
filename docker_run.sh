#!/usr/bin/env bash
# INPUT_DIR=./trufor-clone/test_docker/images
# OUTPUT_DIR=./trufor-clone/test_docker/output

# mkdir -p ${OUTPUT_DIR}
# # docker run --name=trufor --runtime=nvidia --gpus all -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out trufor -gpu 0 -in data/ -out data_out/
# docker run --name=thesis-demo --runtime=nvidia --gpus all -v $(realpath ./):/thesis-demo/ -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out thesis-demo -gpu 0 -in data/ -out data_out/

docker run --name=thesis-demo --runtime=nvidia --gpus all -v $(realpath ./):/thesis-demo/ thesis-demo bash

# # docker system prune -f