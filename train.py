import torch
import mlflow
import argparse
import importlib
import torch.nn as nn
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader
from utils.split_data_cifar10 import get_data
from utils.trainer import Trainer

def create_experiment(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id
    return exp_id

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--calib_size", type=float, default=0.1)
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="resnet")
    parser.add_argument("--model_version", type=str, default="18")
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--exp_name", type=str, default="temp_exp")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cp_method", type=str, default="hinge")
    args = parser.parse_args()

    # Assign vars
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_size = args.val_size
    calib_size = args.calib_size
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    model_name = args.model_name
    model_version = args.model_version
    max_epochs = args.max_epochs
    exp_name = args.exp_name
    alpha = args.alpha
    cp_method = args.cp_method

    exp_id = create_experiment(exp_name)

    with mlflow.start_run(experiment_id=exp_id) as run:
        artifact_path = Path(run.info.artifact_uri.replace("file://", ""))

        mlflow.log_params(vars(args))

        # Init dataset
        # TODO: implement data_name to get_data
        class_dict, train_dataset, val_dataset, test_dataset, calib_dataset = get_data(val_size, calib_size)

        # Init loader
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True)
        calib_loader = DataLoader(calib_dataset, batch_size=eval_batch_size, shuffle=True)
        
        # Init model
        # TODO: write an import function for this
        MODEL = importlib.import_module(f"models.{model_name}")
        net = MODEL.load_model(model_version)

        # Init loss function
        criterion = nn.CrossEntropyLoss()

        # Init optimizer 
        optimizer = optim.SGD(net.parameters(), lr=0.01,
                            momentum=0.9, weight_decay=5e-4)
        
        # Init scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=3
        )

        # Init trainer
        trainer = Trainer(
            train_loader, val_loader, calib_loader, 
            device, net, optimizer, criterion, scheduler, exp_name, class_dict,
            cp_method, artifact_path
            )

        # Train 
        trainer.train(max_epochs)

        # Test
        trainer.test(calib_loader, test_loader, alpha)



