#!/bin/bash

docker build -t sentiment-classifier .

docker run -it -v /home/danya/datasets/CMU-MOSEI/:/data/ sentiment-classifier:latest
