import torch
import mlflow
import numpy as np
import polars as pl
from cp import create_cp
from .base import BaseTrainer

class ConformalTrainer(BaseTrainer):
    def __init__(self, 
                train_loader,
                val_loader,
                calib_loader,
                device, 
                net, optimizer, criterion, scheduler, exp_name, class_dict,
                artifact_path, cp_method, method_args, 
                ):
        super().__init__(
            train_loader, val_loader, device, net, optimizer,
            criterion, scheduler, exp_name, artifact_path
            )
        
        self.calib_loader = calib_loader
        self.class_dict = class_dict
        self.n_classes = len(self.class_dict)
        self.cp_method = cp_method
        self.method_args = method_args
    
    def test(self, calib_loader, test_loader, alpha, r):
        self.net.eval()
        
        # calibration step
        cp = create_cp(self.cp_method, self.method_args, self.device, self.net, alpha, self.n_classes, calib_loader)

        # run test
        running_loss = 0
        correct_ovr = 0
        len_ovr = 0
        correct_class = np.zeros(self.n_classes)
        len_class = np.zeros(self.n_classes)

        # conformal prediction metric
        pred_sizes = {
            "marginal": [],
            "class-cond": []
        }
        coverages = {
            "marginal": [],
            "class-cond": []
        }

        cls_pred_sizes = {
            "marginal": {c: [] for c in range(self.n_classes)},
            "class-cond": {c: [] for c in range(self.n_classes)}
        }
        cls_coverages = {
            "marginal": {c: [] for c in range(self.n_classes)},
            "class-cond": {c: [] for c in range(self.n_classes)}
        }

        # initialize qhat
        cp.calculate_qhat()

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                outputs = self.net(inputs)

                # calculate ovr acc
                _, preds = torch.max(outputs, 1)
                correct_ovr += (preds == targets).sum().item()
                len_ovr += len(preds)

                # calculate class-based acc
                for c in range(self.n_classes):
                    cls_idx = (targets == c).nonzero().reshape(-1)
                    cls_preds = preds[cls_idx]
                    cls_targets = targets[cls_idx]
                    cls_correct = (cls_preds == cls_targets).sum().item()
                    correct_class[c] += cls_correct
                    len_class[c] += len(cls_preds)

                # calculate cp performance
                marg_pred_sets, cond_pred_sets = cp.calculate_pred_set(outputs)

                for cp_type in ["marginal", "class-cond"]:
                    # 1) Assign prediction set 
                    if cp_type == "marginal":
                        pred_sets = marg_pred_sets
                    elif cp_type == "class-cond":
                        pred_sets = cond_pred_sets
                    
                    # 2) Calculate the size of prediction set 
                    pred_sizes[cp_type] += pred_sets.sum(axis=1).cpu().tolist()

                    # 3) Calculate the size of prediction set per class 
                    for c in range(self.n_classes):
                        cls_idx = (targets == c).nonzero().reshape(-1)
                        cls_pred_sets = pred_sets[cls_idx]
                        cls_pred_sizes[cp_type][c] += cls_pred_sets.sum(axis=1).cpu().tolist()

                    # 4) Calculate the coverage 
                    indices = torch.arange(len(outputs), device=outputs.device)
                    coverages[cp_type] += pred_sets[indices, targets].cpu().tolist()

                    # 5) calculate coverage per class
                    for c in range(self.n_classes):
                        cls_idx = (targets == c).nonzero().reshape(-1)
                        cls_pred_sets = pred_sets[cls_idx]
                        cls_coverages[cp_type][c] += cls_pred_sets[:, c].cpu().tolist()

            loss = self.criterion(outputs, targets)

            running_loss += loss.item()

        # log acc related metrics
        test_loss = running_loss / (batch_idx + 1)
        print(f'test loss: {test_loss:.3f}')

        cls_acc = correct_class / (len_class + 1e-8)

        cls_acc_dict = {
            f"test_{k}_acc":cls_acc[idx] for k, idx in self.class_dict.items()
        }

        ovr_test_acc = {
            "test_loss": test_loss,
            "ovr_test_acc": correct_ovr / len_ovr, 
            **cls_acc_dict
        }
        mlflow.log_metrics(ovr_test_acc, step=r)


        # log cp metrics
        mlflow.log_metric("alpha", alpha)
        for cp_type in ["marginal", "class-cond"]:
            # log avg pred size and avg coverage for each type
            mlflow.log_metrics({
                f"avg_{cp_type}_pred_size": sum(pred_sizes[cp_type]) / len(pred_sizes[cp_type]),
                f"avg_{cp_type}_coverage": sum(coverages[cp_type]) / len(coverages[cp_type]),
            }, step=r)

            # log avg cls pred size and avg cls coverage for each type
            mlflow.log_metrics({
                f"cls_{cp_type}_cls_pred_size_{k}": sum(cls_pred_sizes[cp_type][idx]) / len(cls_pred_sizes[cp_type][idx])
                for k, idx in self.class_dict.items()
            }, step=r)

            avg_cls_coverages = {
                f"cls_{cp_type}_cls_coverage_{k}": sum(cls_coverages[cp_type][idx]) / len(cls_coverages[cp_type][idx])
                for k, idx in self.class_dict.items()
            }
            mlflow.log_metrics(avg_cls_coverages, step=r)

            avg_abs_cls_coverage =  np.abs(1 - alpha - np.array(list(avg_cls_coverages.values()))).mean()
            mlflow.log_metric(f"avg_abs_cls_{cp_type}_coverage", avg_abs_cls_coverage, step=r)

        return test_loss