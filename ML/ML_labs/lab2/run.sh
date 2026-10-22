docker build -t lab-fusion .
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  lab-fusion