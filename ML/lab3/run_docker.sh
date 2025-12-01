#!/bin/bash

docker build -t rl-policy-gradient .

# Check if we can use GPU
USE_GPU=false

if command -v nvidia-smi &> /dev/null; then
    # Check if Docker has nvidia runtime
    if docker info 2>/dev/null | grep -qi "runtimes.*nvidia" || \
       docker info 2>/dev/null | grep -qi "nvidia"; then
        USE_GPU=true
    fi
fi

if [ "$USE_GPU" = true ]; then
    echo "Running with GPU support..."
    docker run -it --gpus all rl-policy-gradient:latest
else
    echo "Running on CPU (GPU not available or nvidia-container-toolkit not configured)"
    echo "The code will automatically detect and use the best available device."
    echo ""
    docker run -it rl-policy-gradient:latest
fi
