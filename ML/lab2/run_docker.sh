#!/bin/bash

docker build -t video-classifier .

# docker run -it -v video-classifier:latest
docker run -e VIDEO_PATH=/videos -v /home/danya/datasets/VidTalk/dataset_videos/:/videos video-classifier:latest
