docker build \
-t "rerun/vid2vid" \
-f Dockerfile .
docker run \
-it \
--rm \
-p 9876:9876 \
--gpus 0 \
-v data:/workspace/data \
rerun/vid2vid \
/bin/bash
# python3 main.py
