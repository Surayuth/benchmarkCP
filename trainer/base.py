import os
import torch
import mlflow
import numpy as np

class BaseTrainer:
    def __init__(self, 
                train_loader,
                val_loader,
                device, 
                net, optimizer, 
                criterion, scheduler, exp_name,
                ):
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.net = net.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.exp_name = exp_name

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