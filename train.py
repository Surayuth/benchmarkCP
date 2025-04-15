import torch
import mlflow
import torch.nn as nn
from torch import optim
from pathlib import Path
from torch.utils.data import DataLoader
from datasets import get_data
from trainer import get_cp_trainer
from models import load_model
from cp.utils.cli import create_parser, get_method_params, get_loss_params
from loss_fn import get_loss_fn

def create_experiment(exp_name):
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = mlflow.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id
    return exp_id


if __name__ == "__main__":
    parser = create_parser()
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
    dset_name = args.dset_name
    cp_method = args.cp_method
    repeats = args.repeats
    test_only = args.test_only
    task = args.task
    loss_fn = args.loss_fn

    # Assign method-specific vars
    method_args = get_method_params(args, cp_method)

    # Assign loss-specific vars
    loss_args = get_loss_params(args, loss_fn)

    exp_id = create_experiment(exp_name)

    with mlflow.start_run(experiment_id=exp_id) as run:
        artifact_path = Path(run.info.artifact_uri.replace("file://", ""))
        mlflow.log_params(vars(args) | method_args)

        for r in range(repeats):
            # Init dataset
            class_dict, train_dataset, val_dataset, test_dataset, calib_dataset = get_data(dset_name, val_size, calib_size)

            # Init loader
            if not test_only:
                train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=True)
                
            test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=True)
            calib_loader = DataLoader(calib_dataset, batch_size=eval_batch_size, shuffle=True)
            
            # Init model
            net = load_model(model_name, model_version)

            # Init loss function
            # TODO: write yaml instead of cli as the params grow
            criterion = get_loss_fn(loss_fn, loss_args)

            # Init optimizer 
            optimizer = optim.SGD(net.parameters(), lr=0.01,
                                momentum=0.9, weight_decay=5e-4)
            
            # Init scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3
            )

            # Init trainer
            trainer = get_cp_trainer(task)(
                device, net, optimizer, criterion, scheduler, exp_name, class_dict,
                artifact_path, cp_method, method_args,
            )

            if not test_only:
                # Train 
                trainer.train(train_loader, val_loader, max_epochs)

            # Test
            trainer.test(calib_loader, test_loader, alpha, r)



