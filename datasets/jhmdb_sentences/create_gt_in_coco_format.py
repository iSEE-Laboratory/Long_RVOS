"""
This script converts the ground-truth annotations of the jhmdb.yaml-sentences dataset to COCO format.
This results in a ground-truth JSON file which can be loaded using the pycocotools API.
Note that during evaluation model predictions need to be converted to COCO format as well (check out trainer.py).
"""

import json
import scipy.io
from tqdm import tqdm
from pycocotools.mask import encode, area
import pandas
from os import path
from glob import glob
import random


def get_image_id(video_id, frame_idx):
    image_id = f'v_{video_id}_f_{frame_idx}'
    return image_id

def get_samples_metadata(root_path, generate_new_samples_metadata, distributed):
    samples_metadata_file_path = f'./jhmdb_sentences_samples_metadata_new.json'
    if not generate_new_samples_metadata:  # load existing metadata file
        with open(samples_metadata_file_path, 'r') as f:
            samples_metadata = [tuple(a) for a in json.load(f)]
            return samples_metadata
    print(f'creating jhmdb.yaml-sentences samples metadata...')
    text_annotations = pandas.read_csv(path.join(root_path, 'jhmdb_annotation.txt'))
    assert len(text_annotations) == 928, 'error: jhmdb_annotation.txt is missing one or more samples.'
    text_annotations = list(text_annotations.to_records(index=False))
    used_videos_ids = set([vid_id for vid_id, _ in text_annotations])
    video_frames_folder_paths = sorted(glob(path.join(root_path, 'Rename_Images', '*', '*')))
    video_frames_folder_paths = {p.split('/')[-1]: p for p in video_frames_folder_paths if p.split('/')[-1] in used_videos_ids}
    video_masks_folder_paths = sorted(glob(path.join(root_path, 'puppet_mask', '*', '*', 'puppet_mask.mat')))
    video_masks_folder_paths = {p.split('/')[-2]: p for p in video_masks_folder_paths if p.split('/')[-2] in used_videos_ids}
    samples_metadata = []
    for video_id, text_query in tqdm(text_annotations):
        video_frames_paths = sorted(glob(path.join(video_frames_folder_paths[video_id], '*.png')))
        video_masks_path = video_masks_folder_paths[video_id]
        video_total_masks = len(scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1))
        # some of the last frames in the video may not have masks and thus cannot be used for evaluation
        # so we ignore them:
        video_frames_paths = video_frames_paths[:video_total_masks]
        video_total_frames = len(video_frames_paths)
        chosen_frames_paths = sorted(random.sample(video_frames_paths, 3))  # sample 3 frames randomly
        for frame_path in chosen_frames_paths:
            samples_metadata.append((video_id, frame_path, video_masks_path, video_total_frames, text_query))
    with open(samples_metadata_file_path, 'w') as f:
        json.dump(samples_metadata, f)


def create_jhmdb_sentences_ground_truth_annotations(samples_metadata, dataset_coco_gt_format_path, **kwargs):
    # Note - it is very important to start counting the instance and category ids from 1 (not 0). This is implicitly
    # expected by pycocotools as it is the convention of the original coco dataset annotations.
    categories_dict = [{'id': 1, 'name': 'dummy_class'}]  # dummy class, as categories are not used/predicted in RVOS
    images_dict = []
    annotations_dict = []
    images_set = set()
    instance_id_counter = 1
    for sample_metadata in tqdm(samples_metadata):
        video_id, chosen_frame_path, video_masks_path, _, text_query = sample_metadata

        chosen_frame_idx = int(chosen_frame_path.split('/')[-1].split('.')[0])
        # read the instance masks:
        all_video_masks = scipy.io.loadmat(video_masks_path)['part_mask'].transpose(2, 0, 1)
        # note that to take the center-frame corresponding mask we switch to 0-indexing:
        mask = all_video_masks[chosen_frame_idx - 1]

        image_id = get_image_id(video_id, chosen_frame_idx)
        assert image_id not in images_set, f'error: image id: {image_id} appeared twice'
        images_set.add(image_id)
        images_dict.append({'id': image_id, 'height': mask.shape[0], 'width': mask.shape[1]})

        mask_rle = encode(mask)
        mask_rle['counts'] = mask_rle['counts'].decode('ascii')
        mask_area = float(area(mask_rle))
        instance_annot = {'id': instance_id_counter,
                          'image_id': image_id,
                          'category_id': 1,  # dummy class, as categories are not used/predicted in RVOS
                          'segmentation': mask_rle,
                          'area': mask_area,
                          'iscrowd': 0,
                          }
        annotations_dict.append(instance_annot)
        instance_id_counter += 1
    dataset_dict = {'categories': categories_dict, 'images': images_dict, 'annotations': annotations_dict}
    with open(dataset_coco_gt_format_path, 'w') as f:
        json.dump(dataset_dict, f)

if __name__ == '__main__':
    dataset_path = "data/jhmdb_sentences"
    sample_meta = get_samples_metadata(dataset_path, generate_new_samples_metadata=False, distributed=False)
    subset_type = 'test'
    output_path = f'data/jhmdb_sentences/jhmdb_sentences_{subset_type}_annotations_in_coco_format.json'
    create_jhmdb_sentences_ground_truth_annotations(sample_meta, output_path)

