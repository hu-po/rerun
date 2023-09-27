#!/bin/bash

# Default values
OUTPUT_FILENAME="../data/raw_$(date +%Y%m%d_%H%M%S).mp4"
FRAMERATE=30
VIDEO_SIZE="2048x1080"

# Parse command-line arguments
while (( "$#" )); do
    case "$1" in
        --output_filename)
            OUTPUT_FILENAME=$2
            shift 2
        ;;
        --framerate)
            FRAMERATE=$2
            shift 2
        ;;
        --video_size)
            VIDEO_SIZE=$2
            shift 2
        ;;
        *)
            echo "Error: Invalid argument"
            exit 1
    esac
done

# Check if the output directory exists. If not, create it.
mkdir -p "$(dirname "$OUTPUT_FILENAME")"

# Record video using ffmpeg
start_time=$(date +%s)
ffmpeg -y -f v4l2 -r $FRAMERATE -video_size $VIDEO_SIZE -i /dev/video0 -vf "vflip" -c:v h264 $OUTPUT_FILENAME
end_time=$(date +%s)

# Calculate and print the duration of the recording
duration=$((end_time - start_time))
echo "Recording safely terminated."
echo "Output filename: $OUTPUT_FILENAME"
echo "Length of recording: $duration seconds"
