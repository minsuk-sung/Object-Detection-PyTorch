import os
import torch
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()

default_checkpoint = {
    "epoch": 0,

    # train
    "train_loss": [],
    "train_loss_classifier": [],
    "train_loss_box_reg": [],
    "train_loss_objectness": [],
    "train_loss_rpn_box_reg": [],

    # valid
    "valid_loss": [],
    "valid_loss_classifier": [],
    "valid_loss_box_reg": [],
    "valid_loss_objectness": [],
    "valid_loss_rpn_box_reg": [],

    "lr": [], 
    "model": {},
    "configs":{},
}

def save_checkpoint(checkpoint, dir="./checkpoints", prefix=""):
    # Padded to 4 digits because of lexical sorting of numbers.
    # e.g. 0009.pth
    filename = "{num:0>4}.pth".format(num=checkpoint["epoch"])
    if not os.path.exists(os.path.join(prefix, dir)):
        os.makedirs(os.path.join(prefix, dir))
    torch.save(checkpoint, os.path.join(prefix, dir, filename))

def load_checkpoint(path, cuda=use_cuda):
    if cuda:
        return torch.load(path)
    else:
        # Load GPU model on CPU
        return torch.load(path, map_location=lambda storage, loc: storage)

def init_tensorboard(name="", base_dir="./tensorboard"):
    return SummaryWriter(os.path.join(name, base_dir))

def write_tensorboard(
    writer,
    epoch,
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
):
    writer.add_scalar("train_loss", train_loss, epoch)
    writer.add_scalar("train_loss_classifier", train_loss_classifier, epoch)
    writer.add_scalar("train_loss_box_reg", train_loss_box_reg, epoch)
    writer.add_scalar("train_loss_objectness", train_loss_objectness, epoch)
    writer.add_scalar("train_loss_rpn_box_reg", train_loss_rpn_box_reg, epoch)
    writer.add_scalar("valid_loss", valid_loss, epoch)
    writer.add_scalar("valid_loss_classifier", valid_loss_classifier, epoch)
    writer.add_scalar("valid_loss_box_reg", valid_loss_box_reg, epoch)
    writer.add_scalar("valid_loss_objectness", valid_loss_objectness, epoch)
    writer.add_scalar("valid_loss_rpn_box_reg", valid_loss_rpn_box_reg, epoch)

    for name, param in model.named_parameters():
        writer.add_histogram(
            "{}".format(name), param.detach().cpu().numpy(), epoch
        )
        if param.grad is not None:
            writer.add_histogram(
                "{}/grad".format(name), param.grad.detach().cpu().numpy(), epoch
                )