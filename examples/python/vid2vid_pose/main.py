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

EXAMPLE_DIR: Final = Path(os.path.dirname(__file__))
CACHE_DIR: Final = EXAMPLE_DIR / "cache"


def track_pose(args: argparse.Namespace) -> None:
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

    # Initialize the stable diffusion pipeline
    pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-depth", local_files_only=False, cache_dir=CACHE_DIR.absolute()
    )

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipe = pipe.to("mps")
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")

    pipe.enable_attention_slicing()

    with closing(VideoSource(args.video_path)) as video_source, mp_pose.Pose(
        enable_segmentation=not args.no_segment
    ) as pose:
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

            # Apply style transfer to each frame
            image = Image.fromarray(rgb)

            pipe(
                prompt=args.prompt,
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                negative_prompt=args.n_prompt,
                num_inference_steps=args.num_inference_steps,
                image=image,
            )


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
    parser.add_argument(
        "--prompt",
        type=str,
        default="A tired robot sitting down on a dirt floor. Rusty metal. Unreal Engine. Wall-e",
        help="Positive prompt describing the image you want to generate.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="""
Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
will be used as a starting point, adding more noise to it the larger the `strength`. The number of
denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
be maximum and the denoising process will run for the full number of iterations specified in
`num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
""",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=11,
        help="""
Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
`guidance_scale` is defined as `w` of equation 2. of [Imagen
Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
usually at the expense of lower image quality.
""",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=10,
        help="""
The number of denoising steps. More denoising steps usually lead to a higher quality image at the
expense of slower inference. This parameter will be modulated by `strength`.
""",
    )
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args, "rerun_example_human_pose_tracking")
    track_pose(args.video_path, segment=not args.no_segment, prompt=args.prompt)
    rr.script_teardown(args)


if __name__ == "__main__":
    main()
