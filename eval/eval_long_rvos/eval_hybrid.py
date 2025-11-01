import os
import time
import argparse
import cv2
import json
import numpy as np
from metrics import db_eval_iou, db_eval_boundary
import multiprocessing as mp
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

NUM_WOEKERS = 32

def eval_queue(q, rank, out_dict, pred_path, mask_path):
    while not q.empty():
        # print(q.qsize())
        vid_name, exp = q.get()

        vid = exp_dict[vid_name]

        exp_name = f'{vid_name}_{exp}'

        if not os.path.exists(f'{pred_path}/{vid_name}'):
            print(f'{pred_path}/{vid_name} not found')
            out_dict[exp_name] = [0, 0, 0, 0]
            continue

        first_frame_name = vid['frames'][0]
        pred_0_path = f'{pred_path}/{vid_name}/{exp}/{first_frame_name}.png'
        pred_0 = cv2.imread(pred_0_path, cv2.IMREAD_GRAYSCALE)
        h, w = pred_0.shape

        vid_len = len(vid['frames'])
        gt_masks = np.zeros((vid_len, h, w), dtype=np.uint8)
        pred_masks = np.zeros((vid_len, h, w), dtype=np.uint8)

        obj_id = vid['expressions'][exp]['obj_id']

        for frame_idx, frame_name in enumerate(vid['frames']):
            gt_mask_path = f'{mask_path}/{vid_name}/{obj_id}/{frame_name}.png'
            if os.path.exists(gt_mask_path):
                gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                gt_masks[frame_idx] = gt_mask
            else:
                gt_masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)

            pred_mask_path = f'{pred_path}/{vid_name}/{exp}/{frame_name}.png'
            if os.path.exists(pred_mask_path):
                pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
                pred_masks[frame_idx] = pred_mask
            else:
                pred_masks[frame_idx] = np.zeros((h, w), dtype=np.uint8)


        # sIoU
        j = db_eval_iou(gt_masks, pred_masks)
        f = db_eval_boundary(gt_masks, pred_masks)

        # tIoU & vIoU
        gt_frames = np.any(gt_masks != 0, axis=(1, 2))
        pred_frames = np.any(pred_masks != 0, axis=(1, 2))
        inters_frames = gt_frames & pred_frames
        union = (gt_frames | pred_frames).sum()

        if np.isclose(union, 0):
            tiou = viou = 1
        else:
            tiou = inters_frames.sum() / union
            viou = j[inters_frames].sum() / union

        out_dict[exp_name] = [j.mean(), f.mean(), tiou, viou]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--split", type=str, default="valid")
    args = parser.parse_args()

    exp_path = f'data/long_rvos_split/{args.split}/meta_expressions.json'
    mask_path = f'data/long_rvos_split/{args.split}/Annotations'
    save_name = f'hybrid_{args.split}.json'

    queue = mp.Queue()
    exp_dict = json.load(open(exp_path))['videos']

    shared_exp_dict = mp.Manager().dict(exp_dict)
    output_dict = mp.Manager().dict()

    for vid_name in exp_dict:
        vid = exp_dict[vid_name]
        for exp in vid['expressions']:
            if 'type' not in vid['expressions'][exp]:
                print(vid_name, exp)
            if vid['expressions'][exp]['type'] == "static+dynamic":
                queue.put([vid_name, exp])

    print("Number: ", queue.qsize())

    start_time = time.time()
    processes = []
    for rank in range(NUM_WOEKERS):
        p = mp.Process(target=eval_queue, args=(queue, rank, output_dict, args.pred_path, mask_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    save_name = os.path.join('/'.join(args.pred_path.split('/')[:-1]), save_name)

    with open(save_name, 'w') as f:
        json.dump(dict(output_dict), f)

    j, f, tiou, viou = [], [], [], []

    for v in output_dict.values():
        j.append(v[0])
        f.append(v[1])
        tiou.append(v[2])
        viou.append(v[3])

    print("---- Hybrid Type ----")
    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')
    print(f'tIoU: {np.mean(tiou)}')
    print(f'vIoU: {np.mean(viou)}')

    end_time = time.time()
    total_time = end_time - start_time
    print("time: %.4f s" % (total_time))
