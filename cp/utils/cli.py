import argparse
from cp import _METHODS
from loss_fn import _LOSSFN


def get_method_params(args, method_name):
    """Extract method-specific parameters from args based on the registry."""
    method_params = {}
    
    # Only extract parameters defined for this method
    if method_name in _METHODS:
        for param in _METHODS[method_name]['params']:
            param_name = param['name']
            # Get the value from args
            method_params[param_name] = getattr(args, param_name)
    
    return method_params

def get_loss_params(args, loss_name):
    """Extract loss-specific parameters from args based on the registry."""
    loss_params = {}
    
    # Only extract parameters defined for this method
    if loss_name in _LOSSFN:
        for param in _LOSSFN[loss_name]['params']:
            param_name = param['name']
            # Get the value from args
            loss_params[param_name] = getattr(args, param_name)
    
    return loss_params


def create_parser():
    parser = argparse.ArgumentParser()

    # common arguments
    common_group = parser.add_argument_group('Common Options')

    common_group.add_argument("--val_size", type=float, default=0.1)
    common_group.add_argument("--calib_size", type=float, default=0.1)
    common_group.add_argument("--train_batch_size", type=int, default=64)
    common_group.add_argument("--eval_batch_size", type=int, default=128)
    common_group.add_argument("--model_name", type=str, default="cifar10_resnet")
    common_group.add_argument("--model_version", type=str, default="18")
    common_group.add_argument("--max_epochs", type=int, default=2)
    common_group.add_argument("--exp_name", type=str, default="temp_exp")
    common_group.add_argument("--alpha", type=float, default=0.1)
    common_group.add_argument("--dset_name", type=str, default="cifar10")
    common_group.add_argument("--repeats", type=int, default=1, help="repeats are equivalent to steps in mlflow logs.")
    common_group.add_argument("--test_only", action="store_true", help="test only flag")
    common_group.add_argument("--task", type=str, default="class", help="classification (class) or regression (reg)")
    common_group.add_argument("--loss_fn", type=str, default="ce", help="loss function")

    # available cp methods
    common_group.add_argument('--cp_method', type=str, required=True, 
                             choices=list(_METHODS.keys()), help='CP method to use')

    # Create argument groups for each method
    method_groups = {}
    for method_name in _METHODS.keys():
        method_groups[method_name] = parser.add_argument_group(f'{method_name} Method Options')
        
    # Add method-specific arguments to their respective groups
    for method, info in _METHODS.items():
        group = method_groups[method]
        for param in info['params']:
            group.add_argument(f'--{param["name"]}', type=param['type'], 
                             default=param['default'], 
                             help=param["help"])
            
    # Create argument groups for each loss fn
    loss_groups = {}
    for loss_name in _LOSSFN.keys():
        loss_groups[loss_name] = parser.add_argument_group(f'{loss_name} Method Options')

    # Add loss-specific arguments to their respective groups
    for loss_name, info in _LOSSFN.items():
        group = loss_groups[loss_name]
        for param in info['params']:
            group.add_argument(f'--{param["name"]}', type=param['type'], 
                             default=param['default'], 
                             help=param["help"])
        
    return parser