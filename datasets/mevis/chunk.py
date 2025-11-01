import math
import json
from os import path
from copy import deepcopy

chunk_size = 36


def chunk_expressions(exp_path, save_path):
    new_exp_dict = {"videos": {}}
    exp_dict = json.load(open(exp_path))['videos']
    for exp_id, exp_data in exp_dict.items():
        frames = exp_data['frames']
        N = math.ceil(len(frames) / chunk_size)
        frame_trunks = [[frames[i] for i in range(j, len(frames), N)] for j in range(N)]
        for i, trunk in enumerate(frame_trunks):
            new_exp_dict["videos"][f'{exp_id}_{i}'] = deepcopy(exp_data)
            new_exp_dict["videos"][f'{exp_id}_{i}']['frames'] = trunk
    json.dump(new_exp_dict, open(save_path, 'w'), indent=4)


if __name__ == '__main__':
    for subset_type in ['valid', 'valid_u']:
        dataset_path = f'data/mevis/{subset_type}'
        exp_path = path.join(dataset_path, 'meta_expressions.json')
        save_path = path.join(dataset_path, 'chunked_meta_expressions.json')
        print(f'generating {save_path}')
        chunk_expressions(exp_path, save_path)

