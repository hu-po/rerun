#!/usr/bin/env python3
"""Use the MediaPipe Pose solution to detect and track a human pose in video."""
from __future__ import annotations

import argparse
import logging
import os
import platform
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
import requests
import rerun as rr  # pip install rerun-sdk
import torch
from huggingface_pipeline import StableDiffusionDepth2ImgPipeline
from PIL import Image


def track_pose(video_path: str, segment: bool) -> None:
    mp_pose = mp.solutions.pose

    rr.log_annotation_context(
        "/",
        rr.ClassDescription(
            info=rr.AnnotationInfo(id=0, label="Person"),
            keypoint_annotations=[rr.AnnotationInfo(id=lm.value, label=lm.name) for lm in mp_pose.PoseLandmark],
            keypoint_connections=mp_pose.POSE_CONNECTIONS,
        ),
    )
    # Use a separate annotation context for the segmentation mask.
    rr.log_annotation_context(
        "video/mask",
        [rr.AnnotationInfo(id=0, label="Background"), rr.AnnotationInfo(id=1, label="Person", color=(0, 0, 0))],
    )
    rr.log_view_coordinates("person", up="-Y", timeless=True)

    with closing(VideoSource(video_path)) as video_source, mp_pose.Pose(enable_segmentation=segment) as pose:
        for bgr_frame in video_source.stream_bgr():
            rgb = cv2.cvtColor(bgr_frame.data, cv2.COLOR_BGR2RGB)
            rr.set_time_seconds("time", bgr_frame.time)
            rr.set_time_sequence("frame_idx", bgr_frame.idx)
            rr.log_image("video/rgb", rgb, jpeg_quality=75)

            results = pose.process(rgb)
            h, w, _ = rgb.shape
            landmark_positions_2d = read_landmark_positions_2d(results, w, h)
            if landmark_positions_2d is not None:
                rr.log_points("video/pose/points", landmark_positions_2d, keypoint_ids=mp_pose.PoseLandmark)

            landmark_positions_3d = read_landmark_positions_3d(results)
            if landmark_positions_3d is not None:
                rr.log_points("person/pose/points", landmark_positions_3d, keypoint_ids=mp_pose.PoseLandmark)

            segmentation_mask = results.segmentation_mask
            if segmentation_mask is not None:
                rr.log_segmentation_image("video/mask", segmentation_mask)


def read_landmark_positions_2d(
    results: Any,
    image_width: int,
    image_height: int,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None:
        return None
    else:
        normalized_landmarks = [results.pose_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        return np.array([(image_width * lm.x, image_height * lm.y) for lm in normalized_landmarks])


def read_landmark_positions_3d(
    results: Any,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None:
        return None
    else:
        landmarks = [results.pose_world_landmarks.landmark[lm] for lm in mp.solutions.pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])


@dataclass
class VideoFrame:
    data: npt.NDArray[np.uint8]
    time: float
    idx: int


class VideoSource:
    def __init__(self, path: str):
        self.capture = cv2.VideoCapture(path)

        if not self.capture.isOpened():
            logging.error("Couldn't open video at %s", path)

    def close(self) -> None:
        self.capture.release()

    def stream_bgr(self) -> Iterator[VideoFrame]:
        while self.capture.isOpened():
            idx = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            is_open, bgr = self.capture.read()
            time_ms = self.capture.get(cv2.CAP_PROP_POS_MSEC)

            if not is_open:
                break

            yield VideoFrame(data=bgr, time=time_ms * 1e-3, idx=idx)


def main() -> None:
    # Ensure the logging gets written to stderr:
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.getLogger().setLevel("INFO")

    parser = argparse.ArgumentParser(description="Uses the MediaPipe Pose solution to track a human pose in video.")
    parser.add_argument("--video_path", type=str, default="data/test.mp4", help="Full path to video to run on.")
    parser.add_argument("--no-segment", action="store_true", help="Don't run person segmentation.")
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args, "rerun_example_human_pose_tracking")
    track_pose(args.video_path, segment=not args.no_segment)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
