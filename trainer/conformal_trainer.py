import torch
import mlflow
import numpy as np
import polars as pl
from cp import create_cp
from pathlib import Path
from .base import BaseTrainer

class ConformalTrainer(BaseTrainer):
    def __init__(self, 
                train_loader,
                val_loader,
                calib_loader,
                device, 
                net, optimizer, criterion, scheduler, exp_name, class_dict,
                cp_method,
                artifact_path
                ):
        super().__init__(
            train_loader, val_loader, device, net, optimizer,
            criterion, scheduler, exp_name
            )
        
        self.calib_loader = calib_loader
        self.class_dict = class_dict
        self.n_classes = len(self.class_dict)
        self.cp_method = cp_method
        self.artifact_path = Path(artifact_path)
    
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
        ovr_test_acc = {
            "test_loss": test_loss,
            "ovr_test_acc": correct_ovr / len_ovr, 
        }
        mlflow.log_metrics(ovr_test_acc)

        # log CP related metrics
        cp_report = {
            "avg_pred_size": {
                "marginal": 0,
                "class-cond": 0
            },
            "avg_coverage": {
                "marginal": 0,
                "class-cond": 0
            },
            "cls_pred_size": {
                "marginal": {},
                "class-cond": {}
            },
            "cls_coverage": {
                "marginal": {},
                "class-cond": {}
            }
        }

        mlflow.log_metric("alpha", alpha)
        for cp_type in ["marginal", "class-cond"]:
            # log avg pred size and avg coverage for each type
            mlflow.log_metrics({
                f"avg_{cp_type}_coverage": sum(coverages[cp_type]) / len(coverages[cp_type]),
                f"avg_{cp_type}_pred_size": sum(pred_sizes[cp_type]) / len(pred_sizes[cp_type])
            })
            
            cp_report["avg_pred_size"][cp_type] = sum(pred_sizes[cp_type]) / len(pred_sizes[cp_type])
            cp_report["avg_coverage"][cp_type] = sum(coverages[cp_type]) / len(coverages[cp_type])

            for k, idx in self.class_dict.items():
                cp_report["cls_pred_size"][cp_type][k] = sum(cls_pred_sizes[cp_type][idx]) / len(cls_pred_sizes[cp_type][idx])
                cp_report["cls_coverage"][cp_type][k] = sum(cls_coverages[cp_type][idx]) / len(cls_coverages[cp_type][idx])
        
        
        # Overall acc metrics
        acc_dict = ovr_test_acc | cls_acc_dict
        ovr_acc_stats = pl.DataFrame({
            "stat": list(acc_dict.keys()),
            "values": list(acc_dict.values())
        })

        # Overall CP metrics
        ovr_cp_dict = {
                f"avg_pred_size_{cp_type}": cp_report["avg_pred_size"][cp_type]
                for cp_type in ["marginal", "class-cond"]
            } | {
                f"avg_coverage_{cp_type}": cp_report["avg_coverage"][cp_type]
                for cp_type in ["marginal", "class-cond"]
            }
        
        ovr_cp_dict = pl.DataFrame(
            {
            "stat": ["avg_pred_size", "avg_coverage"],
            } |
            {
                cp_type: [cp_report["avg_pred_size"][cp_type], cp_report["avg_coverage"][cp_type]]
                for cp_type in ["marginal", "class-cond"]
            }
        )
        ovr_cp_stats = pl.DataFrame(ovr_cp_dict)

        # Class-based CP metrics
        avg_cls_pred_size_df = pl.DataFrame(
                {"class": list(self.class_dict.keys())} |
                {cp_type: list(cp_report["cls_pred_size"][cp_type].values()) for cp_type in ["marginal", "class-cond"]}
            )
        
        avg_cls_coverage_df = pl.DataFrame(
                {"class": list(self.class_dict.keys())} | 
                {cp_type: list(cp_report["cls_coverage"][cp_type].values()) for cp_type in ["marginal", "class-cond"]}
            )

        # Export to csv
        ovr_acc_stats.write_csv(self.artifact_path / "ovr_acc_stats.csv")
        ovr_cp_stats.write_csv(self.artifact_path / "ovr_cp_stats.csv")
        avg_cls_pred_size_df.write_csv(self.artifact_path / "avg_cls_pred_size.csv")
        avg_cls_coverage_df.write_csv(self.artifact_path / "avg_cls_coverage.csv")

        return test_loss