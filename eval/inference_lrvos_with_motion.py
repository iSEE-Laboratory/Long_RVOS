import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import sys

import misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json
import gc

from tqdm import tqdm
import shutil

import multiprocessing as mp
import threading
import warnings
warnings.filterwarnings("ignore")

from ruamel.yaml import YAML
from easydict import EasyDict
from torch.cuda.amp import autocast
from misc import nested_tensor_from_videos_list
from models.GroundingDINO.utils import compute_mask

import gc
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

from functools import partial


# colormap
color_list = utils.colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    print("Inference only supports for batch size = 1")
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    save_dir = os.path.join(args.output_dir, args.dataset_name, args.version, split)
    output_dir = os.path.join(save_dir, "Annotations")
    os.makedirs(output_dir, exist_ok=True)
    shutil.copyfile(src=args.config_path, dst=os.path.join(save_dir, 'config.yaml'))

    save_visualize_path_prefix = os.path.join(save_dir, '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    args.dataset_path = "/data1/tianming/long_rvos/data/long_rvos/"
    root = Path(args.dataset_path)
    dataset_path = os.path.join(root, split)
    meta_file = os.path.join(dataset_path, "meta_expressions.json")
    with open(meta_file, "r") as f:
        data = json.load(f)["videos"]
    video_list = list(data.keys())

    # create subprocess
    thread_num = args.num_gpus
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = math.ceil(float(video_num) / float(thread_num))

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]
        p = mp.Process(target=sub_processor, args=(lock, i, args, data,
                                                   output_dir, save_visualize_path_prefix,
                                                   dataset_path, sub_video_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("\n" + "="*50)
    print("Total inference time: %.4f s" % (total_time))
    print(f"Total frames processed: {num_all_frames_gpus}")
    if num_all_frames_gpus > 0 and total_time > 0:
        overall_fps = num_all_frames_gpus / total_time
        print(f"Overall FPS (including data loading): {overall_fps:.4f} fps")
    print("="*50)
    print("\nSave results at: {}".format(output_dir))



def sub_processor(lock, pid, args, data, save_path_prefix, save_visualize_path_prefix, dataset_path, video_list):
    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=len(video_list),
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)
    device = args.device

    sam2_checkpoint = "sam2/checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device, vos_optimized=False)

    # model
    model = build_model(args)[0]
    model.to(device)

    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint["model_state_dict"]
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        del checkpoint
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    num_all_frames = 0
    model.eval()

    total_inference_time = 0.0
    total_frames = 0

    img_folder = os.path.join(dataset_path, "JPEGImages")
    split_name = os.path.basename(dataset_path)
    motion_path = os.path.join("motions", split_name)
    frame_types = json.load(open(os.path.join(motion_path, 'frame_types.json')))

    # 1. For each video
    for video in video_list:
        metas = [] # list[dict], length is number of expressions

        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
            metas.append(meta)
        meta = metas

        # store images
        frames = data[video]["frames"]
        video_name = video
        start_indexs = np.where(np.array(frame_types[video_name]) == 'I')[0]
        end_indexs = start_indexs + 12
        key_frame_ids = start_indexs[:-1][end_indexs[:-1] == start_indexs[1:]]
        key_frame_ids = key_frame_ids.tolist()

        if key_frame_ids[0] > 0:
            key_frame_ids.insert(0, 0)
            padding_start = True
        else:
            padding_start = False

        num_key_frames = len(key_frame_ids)

        imgs = []
        for t in key_frame_ids:
            frame = frames[t]
            img_path = os.path.join(img_folder, video_name, frame + ".jpg")
            img = Image.open(img_path).convert('RGB')
            origin_w, origin_h = img.size
            imgs.append(transform(img))  # list[img]
        imgs = torch.stack(imgs, dim=0).to(args.device)  # [video_len, 3, h, w]
        samples = nested_tensor_from_videos_list(imgs[None], size_divisibility=1)
        img_h, img_w = imgs.shape[-2:]
        size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
        target = {"size": size, "frames_idx": range(video_len)}

        motion_len = args.motion_len
        motions = np.load(os.path.join(motion_path, video + '.npy'))
        motions = torch.from_numpy(motions).to(device)  # T, 2, H/16, W/16

        motion_list = []
        for k in key_frame_ids:
            if k == motions.shape[0]-1 or (padding_start and k == 0):
                motion_list.append(motions[[k]])  # padding
            else:
                motion_list.append(motions[k+1:k+1+motion_len])

        motions = nested_tensor_from_videos_list(motion_list, size_divisibility=1)

        all_masks = {}
        all_boxes = {}

        try:
            # 2. For each expression
            for i in range(num_expressions):
                video_name = meta[i]["video"]
                exp = meta[i]["exp"]
                exp_id = meta[i]["exp_id"]
                frames = meta[i]["frames"]
                
                torch.cuda.synchronize()
                inference_start = time.time()
                with torch.no_grad():
                    with autocast(args.enable_amp):
                        outputs = model.infer(samples, [exp], motions, [target])
                torch.cuda.synchronize()
                inference_end = time.time()
                total_inference_time += (inference_end - inference_start)

                pred_logits = outputs["pred_logits"][0] # [t, q, k]
                pred_masks = outputs["pred_masks"][0]   # [t, q, h, w]
                pred_boxes = outputs["pred_boxes"][0]   # [t, q, 4]

                # according to pred_logits, select the query index
                pred_scores = pred_logits.sigmoid() # [t, q, k]
                pred_scores = pred_scores.mean(0)   # [q, k]
                max_scores, _ = pred_scores.max(-1) # [q,]
                _, max_ind = max_scores.max(-1)     # [1,]
                max_inds = max_ind.repeat(num_key_frames)
                pred_masks = pred_masks[range(num_key_frames), max_inds, ...] # [t, h, w]
                pred_masks = pred_masks.unsqueeze(0)
                pred_boxes = pred_boxes[range(num_key_frames), max_inds] # [t, 4]

                # unpad
                pred_masks = pred_masks[:, :, :img_h, :img_w]
                pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)[0]
                pred_masks = (pred_masks > 0).cpu().numpy()

                all_masks[i] = pred_masks
                all_boxes[i] = pred_boxes

            torch.cuda.empty_cache()

            # for sam2
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                states = video_predictor.init_state(video_path=os.path.join(img_folder, video_name),
                                                    prompt_frame_ids=key_frame_ids, offload_video_to_cpu=True)
                for object_id, masks in all_masks.items():
                    for idx, state in enumerate(states):
                        _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                            inference_state=state,
                            frame_idx=0,
                            obj_id=object_id,
                            mask=masks[idx]
                        )

                torch.cuda.synchronize()
                sam2_start = time.time()
                clip_masks = []
                for idx, state in enumerate(states):
                    for frame_idx, object_ids, out_mask_logits in video_predictor.propagate_in_video(state):
                        mask = out_mask_logits[:, 0] > 0
                        clip_masks.append(mask.cpu())
                torch.cuda.synchronize()
                sam2_end = time.time()
                total_inference_time += (sam2_end - sam2_start)
                
                total_frames += video_len


            clip_masks = torch.stack(clip_masks, dim=0) # T, E, H, W
            clip_masks = clip_masks.transpose(0, 1) # E, T, H, W
            clip_masks = clip_masks.numpy()
            torch.cuda.empty_cache()

            if args.visualize:
                for i in range(num_expressions):
                    exp_id = meta[i]["exp_id"]
                    mask = clip_masks[i]
                    box = all_boxes[i]
                    for t in range(len(frames)):
                        frame = frames[t]
                        # original
                        img_path = os.path.join(img_folder, video_name, frame + '.jpg')
                        source_img = Image.open(img_path).convert('RGBA') # PIL image

                        draw = ImageDraw.Draw(source_img)

                        # draw boxes on key frames
                        if t in key_frame_ids:
                            k = key_frame_ids.index(t)
                            draw_boxes = box[k].unsqueeze(0)
                            draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()
                            xmin, ymin, xmax, ymax = draw_boxes[0]
                            draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[i%len(color_list)]), width=2)

                        # draw mask
                        source_img = vis_add_mask(source_img, mask[t], color_list[i%len(color_list)])

                        # save
                        save_visualize_path_dir = os.path.join(save_visualize_path_prefix, video_name, exp_id)
                        if not os.path.exists(save_visualize_path_dir):
                            os.makedirs(save_visualize_path_dir)
                        save_visualize_path = os.path.join(save_visualize_path_dir, frame + '.png')
                        source_img.save(save_visualize_path)

            # save binary image
            for i in range(num_expressions):
                exp_id = meta[i]["exp_id"]
                masks = clip_masks[i]
                save_path = os.path.join(save_path_prefix, video_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                for t in range(len(frames)):
                    frame = frames[t]
                    mask = masks[t].astype(np.float32)
                    mask = Image.fromarray(mask * 255).convert('L')
                    save_file = os.path.join(save_path, frame + ".png")
                    mask.save(save_file)

        except torch.cuda.OutOfMemoryError as e:
            with open(os.path.join(save_path_prefix, f'oom_{pid}.txt'), 'a') as f:
                f.write(os.path.join(save_path_prefix, video_name) + '\n')
            continue

        with lock:
            progress.update(1)
    
    if total_frames > 0 and total_inference_time > 0:
        fps = total_frames / total_inference_time
        print(f"\n[Processor {pid}] Inference FPS: {fps:.4f} fps "
              f"(Total frames: {total_frames}, Total inference time: {total_inference_time:.4f}s)")
    
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x-10, y, x+10, y), tuple(cur_color), width=4)
        draw.line((x, y-10, x, y+10), tuple(cur_color), width=4)

def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8') # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img

if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Inference')
    parser.add_argument('--config_path', '-c', default='configs/lrvos_swint.yaml',
                        help='path to configuration file')
    parser.add_argument("--checkpoint_path", '-ckpt', required=True,
                        help="The checkpoint path"
                        )
    parser.add_argument('--split', required=True, help='valid or test')
    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--visualize", action='store_true')
    parser.add_argument('--num_gpus', '-ng', type=int, required=True,
                        help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--motion_len", default=3, type=int)
    args = parser.parse_args()
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config.pop("motion_len")
    config = {k: v['value'] for k, v in config.items()}
    args = {**config, **vars(args)}
    args = EasyDict(args)
    print("motion length:",args.motion_len)
    main(args)
