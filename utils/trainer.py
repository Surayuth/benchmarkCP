import os
import torch
import mlflow
import numpy as np
from pathlib import Path
from cp import create_cp

class Trainer:
    def __init__(self, 
                train_loader,
                val_loader,
                calib_loader,
                device, 
                net, optimizer, criterion, scheduler, exp_name, class_dict,
                cp_method,
                artifact_path
                ):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.calib_loader = calib_loader

        self.device = device
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.exp_name = exp_name
        self.artifact_path = None
        self.class_dict = class_dict
        self.n_classes = len(self.class_dict)
        self.cp_method = cp_method
        self.artifact_path = artifact_path

    def _train_one_epoch(self, epoch, print_freq=100):
        self.net.train()

        running_loss = 0
        ovr_loss = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            ovr_loss += loss.item()

            if batch_idx % print_freq == (print_freq - 1):    # print every 2000 mini-batches
                print(f'[{epoch}, {batch_idx + 1:5d}] loss: {running_loss / print_freq:.3f}')
                running_loss = 0.0

        train_loss = ovr_loss / (batch_idx + 1)
        print(f'[{epoch}] train loss: {train_loss:.3f}')
        return train_loss


    def _val_one_epoch(self, epoch):
        self.net.eval()

        running_loss = 0
        for batch_idx, (inputs, targets) in enumerate(self.val_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            running_loss += loss.item()

        val_loss = running_loss / (batch_idx + 1)
        print(f'[{epoch}] val loss: {val_loss:.3f}')
        return val_loss
    
    def train(self, max_epochs):
        min_val_loss = np.inf

        best_epoch = 1
        for epoch in range(1, max_epochs+1):
            train_loss = self._train_one_epoch(epoch)
            val_loss = self._val_one_epoch(epoch)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)

            if val_loss < min_val_loss:
                min_val_loss = val_loss

                # remove previous best model
                prev_model_path = self.artifact_path / f"best_epoch_{best_epoch}.pth"
                if prev_model_path.is_file():
                    os.remove(prev_model_path)

                # save new best model
                best_epoch = epoch
                cur_model_path = self.artifact_path / f"best_epoch_{epoch}.pth"
                torch.save(self.net.state_dict(), cur_model_path)

            self.scheduler.step(val_loss)
    
    def test(self, calib_loader, test_loader, alpha):
        self.net.eval()
        
        # calibration step
        cp = create_cp(self.cp_method, self.device, self.net, alpha, self.n_classes, calib_loader)

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


            loss = self.criterion(outputs, targets)

            running_loss += loss.item()

        test_loss = running_loss / (batch_idx + 1)
        print(f'test loss: {test_loss:.3f}')

        cls_acc = correct_class / (len_class + 1e-8)

        cls_acc_dict = {
            f"test_{k}_acc":cls_acc[idx] for k, idx in self.class_dict.items()
        }

        # log acc related metrics
        mlflow.log_metrics({
            "test_loss": test_loss,
            "ovr_test_acc": correct_ovr / len_ovr, 
            **cls_acc_dict
        })

        # log CP related metrics
        mlflow.log_metric("alpha", alpha)
        for cp_type in ["marginal", "class-cond"]:
            # log avg pred size and avg coverage for each type
            mlflow.log_metrics({
                f"avg_{cp_type}_coverage": sum(coverages[cp_type]) / len(coverages[cp_type]),
                f"avg_{cp_type}_pred_size": sum(pred_sizes[cp_type]) / len(pred_sizes[cp_type])
            })

            # log average pred size per class for each type 
            mlflow.log_metrics({
                f"{cp_type}_cls_pred_size_{k}": sum(cls_pred_sizes[cp_type][idx]) / len(cls_pred_sizes[cp_type][idx]) for k, idx in self.class_dict.items()
            })

            # log average coverage rate per class for each type
            mlflow.log_metrics({
                f"{cp_type}_cls_coverage_{k}": sum(cls_coverages[cp_type][idx]) / len(cls_coverages[cp_type][idx]) for k, idx in self.class_dict.items()
            })

        # TODO: Create conformal prediction report

        return test_loss