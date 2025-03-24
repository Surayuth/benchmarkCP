import os
import torch
import mlflow
import importlib
import numpy as np
from pathlib import Path

class Trainer:
    def __init__(self, 
                train_loader,
                val_loader,
                calib_loader,
                device, 
                net, optimizer, criterion, scheduler, exp_name, class_dict,
                cp_method
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
        exp_id = self.create_experiment(self.exp_name)
        min_val_loss = np.inf

        #   create MLflow session here
        with mlflow.start_run(experiment_id=exp_id) as run:
            self.run_id = run.info.run_id
            self.artifact_path = Path(run.info.artifact_uri.replace("file://", ""))

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
        CP = importlib.import_module(f"cp.{self.cp_method}")
        cp = CP.CP(self.device, self.net, alpha, self.n_classes, calib_loader)

        # run test
        running_loss = 0
        correct_ovr = 0
        len_ovr = 0
        correct_class = np.zeros(self.n_classes)
        len_class = np.zeros(self.n_classes)

        # conformal prediction metric
        pred_sizes = []
        coverages = []

        cls_pred_sizes = {
            c: [] for c in range(self.n_classes)
        }
        cls_coverages = {
            c: [] for c in range(self.n_classes)
        }

        # initialize qhat
        cp.calculate_qhat()

        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            with torch.no_grad():
                outputs = self.net(inputs)

                # calculate prediction set
                pred_sets, cond_pred_sets = cp.calculate_pred_set(outputs)
                
                # 1.1) calculate pred size
                pred_sizes += pred_sets.sum(axis=1).cpu().tolist()

                # 1.2) calculate pred size per class
                for c in range(self.n_classes):
                    cls_idx = (targets == c).nonzero().reshape(-1)
                    cls_pred_sets = pred_sets[cls_idx]

                    cls_pred_sizes[c] += cls_pred_sets.sum(axis=1).cpu().tolist()

                # 2.1) calculate coverage
                indices = torch.arange(len(outputs), device=outputs.device)
                coverages += pred_sets[indices, targets].cpu().tolist()

                # 2.2) calculate coverage per class
                for c in range(self.n_classes):
                    cls_idx = (targets == c).nonzero().reshape(-1)
                    cls_pred_sets = pred_sets[cls_idx]
                    cls_coverages[c] += cls_pred_sets[:, c].cpu().tolist()

                # TODO3:calculate conditional pred size 

                # TODO4:calculate conditional pred size per class 

                # TODO5:calculate conditional coverage 

                # TODO6:calculate conditional coverage per class 


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

        with mlflow.start_run(run_id=self.run_id) as run:
            cls_acc = correct_class / (len_class + 1e-8)

            cls_acc_dict = {
                f"test_{k}_acc":cls_acc[idx] for k, idx in self.class_dict.items()
            }

            marginal_cls_pred_sizes = {
                f"marginal_cls_pred_size_{k}": sum(cls_pred_sizes[idx]) / len(cls_pred_sizes[idx]) for k, idx in self.class_dict.items()
            }

            marginal_cls_coverages = {
                f"marginal_cls_coverage_{k}": sum(cls_coverages[idx]) / len(cls_coverages[idx]) for k, idx in self.class_dict.items()
            }

            marginal_perf = {
                "alpha": alpha,
                "marginal_coverage": sum(coverages) / len(coverages),
                "marginal_pred_size": sum(pred_sizes) / len(pred_sizes),
            }

            mlflow.log_metrics({
                "test_loss": test_loss,
                "ovr_test_acc": correct_ovr / len_ovr,
                **cls_acc_dict,
                **marginal_perf,
                **marginal_cls_pred_sizes,
                **marginal_cls_coverages
            })

            # TODO7: Create conformal prediction report

        return test_loss

    
    def create_experiment(self, exp_name):
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            exp_id = mlflow.create_experiment(exp_name)
        else:
            exp_id = exp.experiment_id
        return exp_id