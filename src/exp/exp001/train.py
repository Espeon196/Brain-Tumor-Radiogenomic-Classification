import argparse
import os
import yaml
import sys
import time
from datetime import datetime, timedelta, timezone
JST = timezone(timedelta(hours=+9), 'JST')
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.model_selection import KFold, StratifiedKFold

import mlflow

from model import BrainTumor2dModel
from dsets import BrainTumor2dSimpleDataset, get_train_transforms, get_valid_transforms
from utils import seed_everything

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description="Train model")
parser.add_argument('--run', help="Run number", type=int, required=True)
parser.add_argument('--input', help="input data folder", type=str, required=True)
args = parser.parse_args()

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(FILE_DIR))
INPUT_DIR = args.input

def log_scaler(name, value, step):
    mlflow.log_metric(name, value)

def prepare_dataloader(df, trn_idx, val_idx, data_path, config):
    train_ = df.loc[trn_idx, :].reset_index(drop=True)
    valid_ = df.loc[val_idx, :].reset_index(drop=True)

    train_ds = BrainTumor2dSimpleDataset(train_, data_path=data_path, transforms=get_train_transforms(), output_label=True)
    valid_ds = BrainTumor2dSimpleDataset(valid_, data_path=data_path, transforms=get_valid_transforms(), output_label=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=config['train_bs'],
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=config['num_workers'],
    )

    val_loader = DataLoader(
        valid_ds,
        batch_size=config['valid_bs'],
        num_workers=config['num_workers'],
        shuffle=False,
        pin_memory=True,
    )
    return train_loader, val_loader

def make_optimizer(params, name, **kwargs):
    # to make Optimizer
    return torch.optim.__dict__[name](params, **kwargs)

def make_scheduler(optimizer, name, **kwargs):
    # to make scheduler
    return torch.optim.lr_scheduler.__dict__[name](optimizer, **kwargs)

def main():
    with open(os.path.join(FILE_DIR, "config/config_{:0=3}.yaml".format(args.run))) as file:
        CONFIG = yaml.safe_load(file)

    base_config = CONFIG['base']
    seed_everything(base_config['seed'])

    df = pd.DataFrame(os.path.join(INPUT_DIR, CONFIG['csv_file']))
    print(INPUT_DIR)

    folds = StratifiedKFold(n_splits=base_config['fold_num'], shuffle=True, random_state=base_config['seed']).split(df, y=df.MGMT_value.tolist())

    model_config = CONFIG.model
    optimizer_config = CONFIG.optimizer
    scheduler_config = CONFIG.scheduler

    for fold, (trn_idx, val_idx) in enumerate(folds):

        model = BrainTumor2dModel(**model_config)
        optimizer = make_optimizer(model.parameters(), **optimizer_config)
        scheduler = make_scheduler(optimizer, **scheduler_config)

        scaler = GradScaler()

        for epoch in range(base_config['epochs']):
            #

if __name__ == "__main__":
    start_time = time.time()
    print('%s: calling main function ... \n' % os.path.basename(__file__))
    main()
    print(f'\nsuccess! [{(time.time()-start_time)/60:.1f} min]')