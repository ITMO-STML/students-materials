#!/bin/bash

cd "$(dirname "$0")"

mkdir -p outputs
mkdir -p ~/.cache/huggingface

docker build -t controlnet-diffusion .

docker run -it --gpus all \
    -v $(pwd)/outputs:/app/outputs \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    controlnet-diffusion:latest python controlnet_diffusion.py "$@"
