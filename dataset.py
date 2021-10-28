import os
import cv2
import time
import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Sampler, Dataset, DataLoader 
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import utils

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class CustomCOCODataset(Dataset):
    def __init__(self, img_path=None, ann_path=None, transforms=None):

        if not img_path or not ann_path:
            raise Exception('You must check your image or annotations path')

        self.img_path = img_path
        self.ann_path = ann_path
        self.transforms = transforms

        self.imgs = sorted(glob.glob(os.path.join(self.img_path, '*.jpg')))[:10]
        self.anns = COCO(self.ann_path)
        self.anns_ids = self.anns.getCatIds()
        self.anns_iscrowd = False

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):

        #########
        # Image #
        #########
        imgId = int(self.imgs[idx].split('/')[-1].split('.')[0])
        img = Image.open(self.imgs[idx]).convert("RGB")

        # This is RGB data
        img_origin = cv2.imread(self.imgs[idx])
        img_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

        ################################################################################
        # Target                                                                       #
        # ['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'] #
        ################################################################################
        annsId = self.anns.getAnnIds(
            imgIds=imgId,
            catIds=self.anns_ids,
            iscrowd=self.anns_iscrowd
        )

        anns = self.anns.loadAnns(annsId)

        targets = {}
        targets["boxes"] = torch.tensor(
            [utils.transform_bbox(ann['bbox']) for ann in anns])
        targets["labels"] = torch.tensor([ann['category_id'] for ann in anns])
        targets["image_id"] = torch.tensor(anns[0]['image_id']) # id vs image_id??
        targets["area"] = torch.tensor([ann['area'] for ann in anns])
        targets["iscrowd"] = torch.tensor([ann['iscrowd'] for ann in anns])

        #############
        # Transform #
        #############
        if self.transforms is not None:
            img = self.transforms(img)

        return img, targets

class CustomCOCOSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

        self.existing_ann = {}
        self.missing_ann = {}
        
        imgIds = {idx:int(path.split('/')[-1].split('.')[0])
                  for idx, path in enumerate(self.data_source.imgs)}

        for idx, imgId in imgIds.items():
            annsId = self.data_source.anns.getAnnIds(
                imgIds=imgId,
                catIds=self.data_source.anns.getCatIds(),
                iscrowd=False,
            )

            anns = self.data_source.anns.loadAnns(annsId)

            if len(anns) <= 0:
                self.missing_ann[idx] = imgId
            else:
                self.existing_ann[idx] = imgId

    def __iter__(self):
        return iter(list(self.existing_ann.keys()))

    def __len__(self):
        return len(self.data_source)

def get_coco_dataset_dataloader(cfg):

    # DATASET
    dataset_train = CustomCOCODataset(
        img_path=cfg.DATA.TRAIN_IMG_PATH,
        ann_path=cfg.DATA.TRAIN_ANN_PATH,
        transforms=T.Compose([
            T.ToTensor(),
            ])
    )

    dataset_valid = CustomCOCODataset(
        img_path=cfg.DATA.VALID_IMG_PATH,
        ann_path=cfg.DATA.VALID_ANN_PATH,
        transforms=T.Compose([
            T.ToTensor(),
            ])
    )

    # SAMPLER
    sampler_train = CustomCOCOSampler(
        data_source=dataset_train
    )

    sampler_valid = CustomCOCOSampler(
        data_source=dataset_valid
    )

    # DATALOADER
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=cfg.BATCH_SIZE,
        shuffle=cfg.SHUFFLE, 
        sampler=sampler_train,
        num_workers=cfg.NUM_WORKERS,  # default: 0
        collate_fn=utils.collate_fn
    )

    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        batch_size=1,  # default: 1 for validation
        shuffle=cfg.SHUFFLE, 
        sampler=sampler_valid,
        num_workers=cfg.NUM_WORKERS,  # default: 0
        collate_fn=utils.collate_fn
    )

    return dataset_train, dataset_valid, dataloader_train, dataloader_valid


if __name__ == '__main__':
    from flags import Flags
    options = Flags('configs/fastrcnn.yaml').get()
    dataset_train, dataset_valid, dataloader_train, dataloader_valid = get_train_valid_dataloader(options)
    print(next(iter(dataloader_train)))
    