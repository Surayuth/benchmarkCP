# datasets/__init__.py
from .cifar10 import get_cifar10_data

# Dictionary-based factory pattern for datasets
_DATASETS = {
    'cifar10': get_cifar10_data,
}

def get_data(dset_name, val_size, calib_size, **kwargs):
    """
    Get dataset splits for a specified dataset.
    
    Args:
        dset_name: String name of the dataset
        val_size: Proportion of data to use for validation
        calib_size: Proportion of data to use for calibration
        **kwargs: Additional dataset-specific arguments
    
    Returns:
        Tuple containing (class_dict, train_dataset, val_dataset, test_dataset, calib_dataset)
    """
    if dset_name not in _DATASETS:
        available = list(_DATASETS.keys())
        raise ValueError(f"Dataset '{dset_name}' not found. Available datasets: {available}")
    
    return _DATASETS[dset_name](val_size, calib_size, **kwargs)