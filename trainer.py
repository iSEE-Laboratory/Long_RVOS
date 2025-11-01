import math
import sys
import os
from os import path
import shutil
import random
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.cuda.amp as amp
from PIL import Image
import gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from util.metrics import calculate_precision_at_k_and_iou_metrics
from util.utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims
from datasets import build_dataset, Collator
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import lr_scheduler
import misc as utils
from models.GroundingDINO.utils import compute_mask
from models import build_model
from rich.progress import track
import json

class Trainer:
    def __init__(self, config, process_id, device_id, num_processes):
        self.config = config

        self.world_size = num_processes
        self.distributed = num_processes > 1
        self.process_id = process_id
        self.is_main_process = process_id == 0
        self.device = init_process_group_and_set_device(num_processes, process_id, device_id, config)

        # fix the seed for reproducibility
        seed = config.seed + config.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion, postprocessor = build_model(config)
        criterion.to(self.device)
        model.to(self.device)
        model_without_ddp = model
        if config.distributed:
            # model = DDP(model, device_ids=[device_id], find_unused_parameters=True)
            model = DDP(model, device_ids=[device_id])
            model_without_ddp = model.module
        self.model = model
        self.criterion = criterion
        self.postprocessor = postprocessor

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        self.dataset_name = config.dataset_name
        if self.dataset_name == 'a2d_sentences' or self.dataset_name == 'jhmdb_sentences':
            self.evaluate = self.evaluate_a2d_sentences
        else:
            self.evaluate = None

        dataset_train = build_dataset(image_set='train', dataset_file=self.dataset_name, **vars(config))
        if self.distributed:
            self.sampler_train = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.rank,
                                                    shuffle=True, seed=config.seed, drop_last=False)
        else:
            self.sampler_train = None
        self.data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=self.sampler_train,
                                            collate_fn=Collator(), num_workers=config.num_workers,
                                            pin_memory=True, shuffle=self.sampler_train is None)
        if self.evaluate is not None:
            dataset_val = build_dataset(image_set='test', dataset_file=self.dataset_name, **vars(config))
            if self.distributed:
                sampler_val = DistributedSampler(dataset_val, num_replicas=config.world_size, rank=config.rank, shuffle=False)
            else:
                sampler_val = None
            eval_batch_size = config.eval_batch_size
            self.data_loader_val = DataLoader(dataset_val, batch_size=eval_batch_size, sampler=sampler_val, drop_last=False,
                                              collate_fn=Collator(), num_workers=config.num_workers,
                                              pin_memory=True)

        # Optimizer, LR-Scheduler, AMP Grad Scaler:
        param_dicts = list(p for p in model_without_ddp.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
        self.num_batches_per_epoch = len(self.data_loader_train)
        if self.dataset_name == 'a2d_sentences':
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.2, verbose=True)
        else:  # refer-youtube-vos:
            self.lr_scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.1, verbose=True)
        self.grad_scaler = amp.GradScaler(enabled=config.enable_amp)
        self.max_norm = config.clip_max_norm

        if self.is_main_process:
            self.output_dir_path = create_output_dir(config)
            self.checkpoint_dir_path = create_checkpoint_dir(self.output_dir_path)
            print(config)
        else:
            self.output_dir_path = ''
        if self.distributed:
            # sync the newly created output dir among all processes:
            output_dir_sync_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(output_dir_sync_list, self.output_dir_path)
            self.output_dir_path = output_dir_sync_list[0]

        self.total_epochs = config.epochs
        self.epoch = 0
        self.iteration = 0
        self.best_mAP = 0
        self.best_loss = math.inf

        if self.config.pretrained_weights is not None:
            print("============================================>")
            print("Load pretrained weights from {} ...".format(self.config.pretrained_weights))
            checkpoint = torch.load(self.config.pretrained_weights, map_location="cpu")
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("============================================>")

    def train(self):
        print("Training started...")
        batch_ob, pg = utils.get_batch_observer(self.epoch, self.total_epochs, len(self.data_loader_train),
                                                disable=(not self.is_main_process) or self.config.debug)
        for self.epoch in range(self.epoch, self.total_epochs):
            self.model.train()
            self.criterion.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            if self.distributed:
                self.sampler_train.set_epoch(self.epoch)
            total_epoch_loss = 0
            loss_sums_dict = {k: 0 for k in self.criterion.weight_dict.keys()}
            batch_ob.start()
            for i, batch_dict in enumerate(self.data_loader_train):
                samples = batch_dict['samples'].to(self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_queries']
                motions = batch_dict['motions'].to(self.device) if 'motions' in batch_dict else None

                with amp.autocast(enabled=self.config.enable_amp):
                    outputs = self.model(samples, text_queries, targets, motions)
                    loss_dict = self.criterion(outputs, targets)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                            k in weight_dict}
                total_loss_reduced = sum(loss_dict_reduced_scaled.values()).item()
                if not math.isfinite(total_loss_reduced):
                    print("Loss is {}, stopping training".format(total_loss_reduced))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                self.grad_scaler.scale(losses).backward()

                # for name, param in self.model.named_parameters():
                #     if param.requires_grad and param.grad is None:
                #         print(name)

                if self.max_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)  # gradients must be unscaled before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                metric_logger.update(loss=total_loss_reduced, **loss_dict_reduced_scaled,)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
                batch_ob.update(pg, advance=1, epoch=self.epoch,
                                loss=metric_logger.meters['loss'].global_avg,
                                cls=metric_logger.meters['loss_ce'].global_avg,
                                bbox=metric_logger.meters['loss_bbox'].global_avg,
                                giou=metric_logger.meters['loss_giou'].global_avg,
                                mask=metric_logger.meters['loss_mask'].global_avg,
                                dice=metric_logger.meters['loss_dice'].global_avg,
                                proj=metric_logger.meters['loss_proj'].global_avg,
                                )
                self.iteration += 1
                total_epoch_loss += total_loss_reduced
                for k in loss_sums_dict.keys():
                    loss_sums_dict[k] += loss_dict_reduced_scaled.get(k, torch.zeros(1)).item()

            batch_ob.reset(pg)
            batch_ob.stop()
            metric_logger.synchronize_between_processes()
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': self.epoch}

            self.lr_scheduler.step()

            # evaluation:
            # run gc collection before starting evaluation to avoid possible OOM errors due to swin-T caching:
            self.clear_memory()
            if self.epoch >= 0 and self.evaluate is not None:
                eval_metrics = self.evaluate()
                self.model.train()  # set model.train() after evaluation to save lora in checkpoint
                for key, value in eval_metrics.items():
                    log_stats['evaluate' + key] = value

            if self.is_main_process:
                if self.dataset_name == 'a2d_sentences':
                    mAP_score = eval_metrics.get('mAP 0.5:0.95')
                    self.save_checkpoint(mAP_score)
                else:
                    self.save_checkpoint(total_epoch_loss)
                with open(os.path.join(self.output_dir_path, 'log.txt'), 'a')as f:
                    f.write(json.dumps(log_stats) + "\n")

            # run gc collection before starting a new epoch to avoid possible OOM errors due to swinT caching :
            self.clear_memory()
            if self.distributed:
                dist.barrier()

    @torch.no_grad()
    def evaluate_a2d_sentences(self):
        self.model.eval()
        predictions = []
        for batch_dict in track(self.data_loader_val, description="Evaluating",
                                disable=(not self.is_main_process) or self.config.debug):
            samples = batch_dict['samples'].to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            text_queries = batch_dict['text_queries']

            outputs = self.model(samples, text_queries, targets)
            outputs.pop('aux_outputs', None)

            processed_outputs = self.postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                   resized_sample_sizes=[t['size'] for t in targets],
                                                   orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]

            for p, image_id in zip(processed_outputs, image_ids):
                for s, m in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})

        if self.distributed:
            # gather and merge predictions from all processes:
            gathered_pred_lists = utils.all_gather(predictions)
            predictions = [p for p_list in gathered_pred_lists for p in p_list]
        eval_metrics = {}
        if self.is_main_process:
            coco_gt = COCO(self.config.dataset_coco_gt_format_path)
            coco_pred = coco_gt.loadRes(predictions)
            coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
            coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
            ap_metrics = coco_eval.stats[:6]
            eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
            if self.config.calculate_precision_and_iou_metrics:
                precision_at_k, overall_iou, mean_iou = (
                    calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred))
                eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
                eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
            print(eval_metrics)
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        return eval_metrics

    def to_device(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [self.to_device(s) for s in sample]
        return sample

    def load_checkpoint(self, checkpoint_path, total_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1  # the epoch after the one saved is about to begin
        if total_epoch == None:
            self.total_epochs = checkpoint['total_epochs']
        else:
            self.total_epochs = total_epoch
        if self.dataset_name == 'a2d_sentences':
            self.best_mAP = checkpoint['best_mAP']
        else:  # refer-youtube-vos
            self.best_loss = checkpoint['best_loss']
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    def save_checkpoint(self, epoch_score):
        if not self.is_main_process:
            return
        is_best = False
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_dict = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        if self.dataset_name == 'a2d_sentences':
            is_best_mAP = epoch_score > self.best_mAP
            if is_best_mAP:
                self.best_mAP = epoch_score
                is_best = True
            checkpoint_dict['best_mAP'] = self.best_mAP
        else:  # refer-youtube-vos
            is_best_loss = epoch_score < self.best_loss
            if is_best_loss:
                self.best_loss = epoch_score
                is_best = True
            checkpoint_dict['best_loss'] = self.best_loss
        filename = self.get_checkpoint_filename()
        torch.save(checkpoint_dict, filename)
        print(f'saved checkpoint: {filename}')
        if is_best:
            best_filename = self.get_checkpoint_filename(is_best=True)
            shutil.copyfile(filename, best_filename)
        # self.remove_extra_checkpoints()

    def get_checkpoint_filename(self, is_best=False):
        basename = 'best' if is_best else f'{self.epoch:02d}'
        return os.path.join(self.checkpoint_dir_path, f'{basename}.pth.tar')

    def remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self.checkpoint_dir_path))
        max_num_checkpoints = 5
        num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self.checkpoint_dir_path, filename))

    def clear_memory(self):
        compute_mask.cache_clear()  # empty cache of SwinT
        gc.collect()
        torch.cuda.empty_cache()


def pre_trained_model_to_finetune(checkpoint, args):
    checkpoint = checkpoint['model_state_dict']
    # only delete the class_embed since the finetuned dataset has different num_classes
    num_layers = args.DeformTransformer['dec_layers'] + 1 if args.DeformTransformer['two_stage'] else args.DeformTransformer['dec_layers']
    for l in range(num_layers):
        del checkpoint["class_embed.{}.weight".format(l)]
        del checkpoint["class_embed.{}.bias".format(l)]

    return checkpoint


def init_process_group_and_set_device(world_size, process_id, device_id, config):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    config.world_size = world_size
    config.rank = process_id
    if device_id != 'cpu':
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample