import os
import argparse
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str)
    parser.add_argument("--split", type=str, default="valid")
    args = parser.parse_args()

    dynamic = json.load(open(os.path.join('/'.join(args.pred_path.split('/')[:-1]), f'dynamic_{args.split}.json')))
    static = json.load(open(os.path.join('/'.join(args.pred_path.split('/')[:-1]), f'static_{args.split}.json')))
    hybrid = json.load(open(os.path.join('/'.join(args.pred_path.split('/')[:-1]), f'hybrid_{args.split}.json')))

    j, f, tiou, viou = [], [], [], []

    for output_dict in [dynamic, static, hybrid]:
        for v in output_dict.values():
            j.append(v[0])
            f.append(v[1])
            tiou.append(v[2])
            viou.append(v[3])

    print("---- Overall Type ----")
    print(f'J: {np.mean(j)}')
    print(f'F: {np.mean(f)}')
    print(f'J&F: {(np.mean(j) + np.mean(f)) / 2}')
    print(f'tIoU: {np.mean(tiou)}')
    print(f'vIoU: {np.mean(viou)}')
