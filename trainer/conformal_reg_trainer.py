import torch
import mlflow
import numpy as np
import polars as pl
from tqdm import tqdm
from cp import create_cp
from .base import BaseTrainer

class ConformalRegTrainer(BaseTrainer):
    def __init__(self,
                device, 
                net, optimizer, criterion, scheduler, exp_name, class_dict,
                artifact_path, cp_method, method_args, 
                ):
        self.class_dict = class_dict
        self.n_classes = len(set(self.class_dict.values()))
        self.cp_method = cp_method
        self.method_args = method_args

    def test(self, calib_loader, test_loader, alpha, r):
        self.net.eval()

        # calibration step
        cp = create_cp(self.cp_method, self.method_args, self.device, self.net, alpha, self.n_classes)

        # run test
        running_loss = 0

        # conformal prediction metric
        pred_sizes = {
            "marginal": [],
            "class-cond": []
        }
        coverages = {
            "marginal": [],
            "class-cond": []
        }

        # initialize qhat 
        # TODO: calculate qhat for reg
        cp.calculate_qhat(calib_loader)