import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
from trainer import Trainer
from pretrainer import PreTrainer

import os

from ruamel.yaml import YAML
from easydict import EasyDict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '12355'


def run(process_id, args):
    with open(args.config_path) as f:
        yaml = YAML(typ='safe', pure=True)
        config = yaml.load(f)
    config = {k: v['value'] for k, v in config.items()}
    config = {**config, **vars(args)}
    config = EasyDict(config)

    for lr in ['lr', 'lr_base']:
        if lr in config.keys():
            config[lr] *= args.batch_size * args.num_devices / 2

    if config.running_mode == 'pretrain':
        trainer = PreTrainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)
    else:
        trainer = Trainer(config, process_id, device_id=args.device_ids[process_id], num_processes=args.num_devices)

    if config.eval:
        print("Load checkpoint from {} ...".format(config.checkpoint_path))
        model_state_dict = torch.load(config.checkpoint_path, map_location=torch.device('cpu'))
        if 'model_state_dict' in model_state_dict.keys():
            model_state_dict = model_state_dict['model_state_dict']
        from torch.nn.parallel import DistributedDataParallel as DDP
        model_without_ddp = trainer.model.module if isinstance(trainer.model, DDP) else trainer.model
        model_without_ddp.load_state_dict(model_state_dict, strict=False)
        trainer.evaluate()
    else:
        if config.checkpoint_path is not None:
            trainer.load_checkpoint(config.checkpoint_path, total_epoch=config.epochs)
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RVOS DINO: Training')
    parser.add_argument('--config_path', '-c', required=True,
                        help='path to configuration file')
    parser.add_argument('--running_mode', '-rm', choices=['train', 'pretrain'], required=True)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--epochs', type=int, default=10,
                        help="the total epochs")
    parser.add_argument("--checkpoint_path", '-ckpt', default=None,
                        help="The checkpoint path"
                        )
    parser.add_argument("--pretrained_weights", '-pw', default=None,
                        help='The pretrained weights path'
                        )
    parser.add_argument("--version", default="refer_dino",
                        help="the saved ckpt and output version")
    parser.add_argument("--lr_drop", default=[8], type=int, nargs='+')
    parser.add_argument('--batch_size', '-bs', type=int, default=2,
                        help='training batch size per device')
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--eval_off', action="store_true")

    gpu_params_group = parser.add_mutually_exclusive_group(required=True)
    gpu_params_group.add_argument('--num_gpus', '-ng', type=int, default=argparse.SUPPRESS,
                                  help='number of CUDA gpus to run on. mutually exclusive with \'gpu_ids\'')
    gpu_params_group.add_argument('--gpu_ids', '-gids', type=int, nargs='+', default=argparse.SUPPRESS,
                                  help='ids of GPUs to run on. mutually exclusive with \'num_gpus\'')
    gpu_params_group.add_argument('--cpu', '-cpu', action='store_true', default=argparse.SUPPRESS,
                                  help='run on CPU. Not recommended, but could be helpful for debugging if no GPU is'
                                       'available.')
    args = parser.parse_args()

    if hasattr(args, 'num_gpus'):
        args.num_devices = max(min(args.num_gpus, torch.cuda.device_count()), 1)
        args.device_ids = list(range(args.num_gpus))
    elif hasattr(args, 'gpu_ids'):
        for gpu_id in args.gpu_ids:
            assert 0 <= gpu_id <= torch.cuda.device_count() - 1, \
                f'error: gpu ids must be between 0 and {torch.cuda.device_count() - 1}'
        args.num_devices = len(args.gpu_ids)
        args.device_ids = args.gpu_ids
    else:  # cpu
        args.device_ids = ['cpu']
        args.num_devices = 1

    if args.num_devices > 1:
        torch.multiprocessing.spawn(run, nprocs=args.num_devices, args=(args,))
    else:  # run on a single GPU or CPU
        run(process_id=0, args=args)
