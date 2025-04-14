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
        super().__init__(
            device, net, optimizer,
            criterion, scheduler, exp_name, artifact_path
            )

        self.class_dict = class_dict
        self.n_classes = -1 # for compatibility
        self.cp_method = cp_method
        self.method_args = method_args

    def test(self, calib_loader, test_loader, alpha, r):
        self.net.eval()

        # calibration step
        cp = create_cp(self.cp_method, self.method_args, self.device, self.net, alpha, self.n_classes)

        # run test
        running_loss = 0

        # conformal prediction metric
        pred_sizes = []
        coverages = []

        # initialize qhat 
        # TODO: calculate qhat for reg
        cp.calculate_qhat(calib_loader)

        for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader),
                                                 total=len(test_loader),
                                                 desc="Calculating test metrics",
                                                 bar_format='{l_bar}{bar:50}{r_bar}{bar:-50b}'
                                                 ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                outputs = self.net(inputs)

                # calculate loss
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()

                # calculate cp performance
                pred_sets = cp.calculate_pred_set(outputs)
                
                lower_bounds = pred_sets[:, 0]
                upper_bounds = pred_sets[:, 1]

                # calculate pred size
                pred_sizes += (upper_bounds - lower_bounds).cpu().tolist()

                # calculate coverage
                coverages += ((lower_bounds <=  targets) & (targets <= upper_bounds)).cpu().tolist()

        test_loss = running_loss / (batch_idx + 1)
        print(f'test loss: {test_loss:.3f}')

        # log test metrics
        ovr_test_acc = {
            "test_loss": test_loss,
        }
        mlflow.log_metrics(ovr_test_acc, step=r)

        # log cp metrics
        avg_pred_size = np.mean(pred_sizes)
        avg_coverage = np.mean(coverages)
        mlflow.log_metrics({
            "avg_pred_size": avg_pred_size,
            "avg_coverage": avg_coverage
        }, step=r)

        

