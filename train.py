"""
nohup python train.py --config_file=configs/faster-rcnn.yaml > train.out &
nohup tensorboard --logdir=./log/ --host=163.152.51.111 --port=6006 > tensorboard.out &
"""

import os
import cv2
import sys
import glob
import fire
import time
import math
import yaml
import random
import shutil
import datetime
from psutil import virtual_memory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T
from torch.utils.data import Sampler, Dataset, DataLoader 
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import utils
from flags import Flags
from checkpoint import (
    default_checkpoint,
    load_checkpoint,
    save_checkpoint,
    init_tensorboard,
    write_tensorboard,
)
from utils import get_network, get_optimizer
from dataset import get_coco_dataset_dataloader

import torchvision.models.detection.mask_rcnn

from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator

def run_one_epoch(cfg, model, optimizer, lr_scheduler, data_loader, epoch, mode='Train'):

    # now = str(re.sub('[^0-9]', '', str(datetime.datetime.now())))

    model.train()    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = '\033[31m{} Epoch: {}\033[00m'.format(mode, epoch)

    lr_scheduler = None

    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, targets in metric_logger.log_every(data_loader, cfg.PRINT_FREQ, header):
        images = list(image.to(cfg.DEVICE) for image in images)
        targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if mode == 'Train':

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    if mode == 'Train':
        lr_scheduler.step()

    return metric_logger

@torch.no_grad()
def evaluate(cfg, model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = utils._get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        # targets = [{k: v.to(cfg.DEVICE) for k, v in t.items()} for t in targets]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        model_time = time.time()
        loss_dict = model(images)
        # print(loss_dict)
        
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in loss_dict]

        # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        # metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    
    return metric_logger

def main(config_file):
    """
    Train math formula recognition model
    """
    cfg = Flags(config_file).get()

    # Set random seed
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    os.environ["PYTHONHASHSEED"] = str(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    is_cuda = torch.cuda.is_available()
    print("--------------------------------")
    print("\033[31mRunning {} on device {}\033[00m\n".format(cfg.NETWORK, cfg.DEVICE))

    # Print system environments
    current_device = torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'
    num_gpus = torch.cuda.device_count()
    num_cpus = os.cpu_count()
    mem_size = virtual_memory().available // (1024 ** 3)
    torch.cuda.empty_cache()
    print(
        "\033[31m[+] System environments\033[00m\n",
        "Device: {}\n".format(torch.cuda.get_device_name(current_device)),
        "Random seed : {}\n".format(cfg.SEED),
        "The number of gpus : {}\n".format(num_gpus),
        "The number of cpus : {}\n".format(num_cpus),
        "Memory Size : {}G\n".format(mem_size),
    )

    ####################################
    # Load checkpoint and print result #
    ####################################
    checkpoint = (
        load_checkpoint(cfg.CHECKPOINT, cuda=is_cuda)
        if cfg.CHECKPOINT != ""
        else default_checkpoint
    )

    ############
    # Get data #
    ############
    dataset_train, dataset_valid, dataloader_train, dataloader_valid = get_coco_dataset_dataloader(cfg)
    print(
        "\033[31m[+] Data Desription\033[00m\n",
        "Train img path : {}\n".format(cfg.DATA.TRAIN_IMG_PATH),
        "Train ann path : {}\n".format(cfg.DATA.TRAIN_ANN_PATH),
        "Valid img path : {}\n".format(cfg.DATA.VALID_IMG_PATH),
        "Valid ann path : {}\n".format(cfg.DATA.VALID_ANN_PATH),
        "Batch size : {}\n".format(cfg.BATCH_SIZE),
        "The number of train samples : {:,}\n".format(len(dataset_train)),
        "The number of valid samples : {:,}\n".format(len(dataset_valid)),
    )

    ###################
    # Get loss, model #
    ###################
    model = get_network(cfg)
    model_state = checkpoint.get("model")
    if model_state:
        model.load_state_dict(model_state)
        print(
        "\033[31m[+] Checkpoint\033[00m\n",
        "Resuming from epoch : {}\n".format(checkpoint["epoch"]),
        "Train Accuracy : {:.5f}\n".format(checkpoint["train_accuracy"][-1]),
        "Train Loss : {:.5f}\n".format(checkpoint["train_losses"][-1]),
        "Valid Accuracy : {:.5f}\n".format(checkpoint["valid_accuracy"][-1]),
        "Valid Loss : {:.5f}\n".format(checkpoint["valid_losses"][-1]),
        )
    
    params_to_optimise = [
        param for param in model.parameters() if param.requires_grad
    ]
    print(
        "\033[31m[+] Network\033[00m\n",
        "Type: {}\n".format(cfg.NETWORK.NAME),
        "Model parameters: {:,}\n".format(
            sum(p.numel() for p in params_to_optimise),
        ),
    )

    #################
    # Get optimizer #
    #################
    optimizer = get_optimizer(params_to_optimise, cfg)
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)
    for param_group in optimizer.param_groups:
        param_group["initial_lr"] = cfg.OPTIMIZER.LR

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg.LR_SCHEDULER.STEP_SIZE, 
        gamma=cfg.LR_SCHEDULER.GAMMA
    )

    print(
        "\033[31m[+] Optimizer\033[00m\n",
        "Type: {}\n".format(cfg.OPTIMIZER.TYPE),
        "Learning rate: {:,}\n".format(cfg.OPTIMIZER.LR),
        "Weight Decay: {:,}\n".format(cfg.OPTIMIZER.WEIGHT_DECAY_RATE),
    )

    #######
    # Log #
    #######
    if not os.path.exists(cfg.PREFIX):
        os.makedirs(cfg.PREFIX)
    log_file = open(os.path.join(cfg.PREFIX, "log.txt"), "w")
    shutil.copy(config_file, os.path.join(cfg.PREFIX, "train_config.yaml"))

    if cfg.PRINT_EPOCHS is None:
        cfg.PRINT_EPOCHS = cfg.NUM_EPOCHS
        
    writer = init_tensorboard(name=cfg.PREFIX.strip("-"))
    start_epoch = checkpoint["epoch"]

    train_loss = checkpoint["train_loss"]
    train_loss_classifier = checkpoint["train_loss_classifier"]
    train_loss_box_reg = checkpoint["train_loss_box_reg"]
    train_loss_objectness = checkpoint["train_loss_objectness"]
    train_loss_rpn_box_reg = checkpoint["train_loss_rpn_box_reg"]

    valid_loss = checkpoint["valid_loss"]
    valid_loss_classifier = checkpoint["valid_loss_classifier"]
    valid_loss_box_reg = checkpoint["valid_loss_box_reg"]
    valid_loss_objectness = checkpoint["valid_loss_objectness"]
    valid_loss_rpn_box_reg = checkpoint["valid_loss_rpn_box_reg"]

    learning_rates = checkpoint["lr"]

    ###############
    # Train model #
    ###############
    valid_early_stop = 0
    valid_best_loss = float('inf')

    for epoch in range(cfg.NUM_EPOCHS):

        # TRAIN
        metric_logger_train = run_one_epoch(cfg, model, optimizer, lr_scheduler, dataloader_train, epoch, mode='Train')
        train_loss.append(metric_logger_train.meters['loss'].value)
        train_loss_classifier.append(metric_logger_train.meters['loss_classifier'].value)
        train_loss_box_reg.append(metric_logger_train.meters['loss_box_reg'].value)
        train_loss_objectness.append(metric_logger_train.meters['loss_objectness'].value)
        train_loss_rpn_box_reg.append(metric_logger_train.meters['loss_rpn_box_reg'].value)

        # EVAL
        metric_logger_valid = run_one_epoch(cfg, model, optimizer, lr_scheduler, dataloader_train, epoch, mode='Valid')
        valid_loss.append(metric_logger_valid.meters['loss'].value)
        valid_loss_classifier.append(metric_logger_valid.meters['loss_classifier'].value)
        valid_loss_box_reg.append(metric_logger_valid.meters['loss_box_reg'].value)
        valid_loss_objectness.append(metric_logger_valid.meters['loss_objectness'].value)
        valid_loss_rpn_box_reg.append(metric_logger_valid.meters['loss_rpn_box_reg'].value)

        evaluate(cfg, model, dataloader_valid, device=cfg.DEVICE)

        ###################################
        # Save checkpoint and make config #
        ###################################
        with open(config_file, 'r') as f:
            option_dict = yaml.safe_load(f)

        save_checkpoint(
            {
                "epoch": start_epoch + epoch + 1,

                "train_loss": train_loss,
                "train_loss_classifier": train_loss_classifier,
                "train_loss_box_reg": train_loss_box_reg,
                "train_loss_objectness": train_loss_objectness,
                "train_loss_rpn_box_reg": train_loss_rpn_box_reg,

                "valid_loss": valid_loss,
                "valid_loss_classifier": valid_loss_classifier,
                "valid_loss_box_reg": valid_loss_box_reg,
                "valid_loss_objectness": valid_loss_objectness,
                "valid_loss_rpn_box_reg": valid_loss_rpn_box_reg,

                "lr": learning_rates,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "configs": option_dict,
            },
            prefix=cfg.PREFIX,
        )

        # log_file.write(output_string + "\n")

        write_tensorboard(
            writer,
            start_epoch + epoch + 1,
            
            train_loss,
            train_loss_classifier,
            train_loss_box_reg,
            train_loss_objectness,
            train_loss_rpn_box_reg,

            valid_loss,
            valid_loss_classifier,
            valid_loss_box_reg,
            valid_loss_objectness,
            valid_loss_rpn_box_reg,

            model,
        )



if __name__ == "__main__":
    fire.Fire(main)