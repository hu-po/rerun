export DATA_PATH="/home/oop/dev/simicam/data"
export CKPT_PATH="/home/oop/dev/simicam/ckpt"
export LOGS_PATH="/home/oop/dev/simicam/logs"
docker build \
-t "rerun/vid2vid" \
-f Dockerfile .
docker run \
-it \
--rm \
-p 9876:9876 \
--gpus 0 \
-v ${DATA_PATH}:/workspace/data \
-v ${CKPT_PATH}:/workspace/ckpt \
-v ${LOGS_PATH}:/workspace/logs \
rerun/vid2vid \
/bin/bash
# python3 main.py
