#!/usr/bin/env bash
docker run --name=thesis-demo --runtime=nvidia --gpus all -v $(realpath ./):/thesis-demo/ thesis-demo bash
