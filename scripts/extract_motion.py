import re
import os
import cv2
import math
import h5py
import json
import random
import shutil
import argparse
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from mvextractor.videocap import VideoCap
import subprocess
import tempfile


def down_sample(motions: torch.Tensor, block_size: int=16):
    h, w = motions.shape[-2:]
    pad_h_size = math.ceil(h / block_size) * block_size
    pad_w_size = math.ceil(w / block_size) * block_size

    padding_h = pad_h_size - h
    padding_w = pad_w_size - w

    pad = (0, padding_w, 0, padding_h)
    motions = torch.nn.functional.pad(motions, pad, mode="constant", value=-10000)
    motions = torch.nn.functional.max_pool2d(motions, kernel_size=block_size, stride=block_size)
    motions[motions < -1000] = 0.0
    return motions


def extract_motions(video_path, rescale=False):
    # The motion vector mean and std
    mean = np.array([[0.0, 0.0]], dtype=np.float64)
    std = np.array([[0.0993703, 0.1130276]], dtype=np.float64)

    # Rescale for the input of motion tokenizer
    if rescale:
        std = std / 10.0

    # Load motion vector from raw video
    cap = VideoCap()
    success = cap.open(video_path)
    if not success:
        print("failed to open {0}".format(video_path))

    motions, frame_types = [], []

    while True:
        ret, frame, motion_vectors, frame_type, timestamp = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        mv = np.ones((h, w, 2)) * -10000  # -10000 indicates padding values
        position = motion_vectors[:, 5:7].clip((0, 0), (w - 1, h - 1))
        mvs = motion_vectors[:, 0:1] * motion_vectors[:, 7:9] / motion_vectors[:, 9:]

        # Normalize the motion vector with resoultion
        mvs[:, 0] = mvs[:, 0] / w
        mvs[:, 1] = mvs[:, 1] / h
        # Normalize the motion vector
        mvs = (mvs - mean) / std

        mv[position[:, 1], position[:, 0]] = mvs
        motions.append(mv)
        frame_types.append(frame_type)
    return motions, frame_types


def _build_temp_video_from_jpegs(frames_dir: str, work_dir: str) -> str:
    """Pack ordered JPEG frames into a temporary mp4 using ffmpeg.

    Returns the path to the generated mp4 inside work_dir.
    """
    # Collect and sort frame files
    frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg')]
    frame_files.sort()
    if len(frame_files) == 0:
        raise RuntimeError(f"No JPEG frames found in {frames_dir}")

    list_path = os.path.join(work_dir, 'list.txt')
    with open(list_path, 'w', encoding='utf-8') as f:
        for fp in frame_files:
            # Use concat demuxer; one frame per image
            f.write(f"file '{fp.replace('\\\\', '/')}'\n")

    out_path = os.path.join(work_dir, 'temp.mp4')
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', list_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        out_path
    ]
    # Run ffmpeg quietly but capture errors
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"ffmpeg failed to build video from {frames_dir}: {result.stderr[:500]}")
    return out_path


def process_single_video(args):
    frames_dir, save_path = args
    try:
        video_name = os.path.basename(frames_dir)

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_video_path = _build_temp_video_from_jpegs(frames_dir, tmpdir)
            motions, frame_types = extract_motions(temp_video_path)

        if not motions:
            return None

        motions = [torch.from_numpy(motion.transpose((2, 0, 1))) for motion in motions]
        motions = torch.stack(motions).float()  # T, C, H, W
        motions = down_sample(motions).numpy()

        save_path = os.path.join(save_path, f"{video_name}.npy")
        np.save(save_path, motions)
        return (video_name, frame_types)

    except Exception as e:
        print(f"Error processing {frames_dir}: {str(e)}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract motion vectors from videos")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/long_rvos",
        help="Root directory of the dataset (default: data/long_rvos)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="motions",
        help="Output directory for motion vectors (default: motions)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "valid", "test"],
        help="Splits to process (default: train valid test)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker processes (default: cpu_count - 2)"
    )
    
    
    args = parser.parse_args()
    
    NUM_WORKERS = args.num_workers if args.num_workers is not None else os.cpu_count() - 2
    
    for split in args.splits:
        dataset_path = os.path.join(args.data_dir, split)
        
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} does not exist, skipping {split}")
            continue

        output_path = os.path.join(args.output_dir, split)
        os.makedirs(output_path, exist_ok=True)
        
        motions_path = os.path.join(output_path, "motions")
        os.makedirs(motions_path, exist_ok=True)

        jpeg_images_dir = os.path.join(dataset_path, "JPEGImages")
        if not os.path.exists(jpeg_images_dir):
            print(f"Warning: {jpeg_images_dir} does not exist, skipping {split}")
            continue

        # Build tasks from per-sequence JPEG folders
        seq_dirs = [os.path.join(jpeg_images_dir, d) for d in os.listdir(jpeg_images_dir) if os.path.isdir(os.path.join(jpeg_images_dir, d))]

        with Pool(processes=NUM_WORKERS) as pool:
            tasks = [(sd, motions_path) for sd in seq_dirs]
            results = list(tqdm(
                pool.imap(process_single_video, tasks),
                total=len(tasks),
                desc=f"Processing {split} videos"
            ))

        frame_types_dict = {}
        for result in results:
            if result is not None:
                video_name, frame_types = result
                frame_types_dict[video_name] = frame_types
        json.dump(frame_types_dict, open(os.path.join(output_path, "frame_types.json"), "w"))