from .base import BaseTrainer
from .conformal_class_trainer import ConformalClassTrainer
from .conformal_reg_trainer import ConformalRegTrainer

__all__ = ["BaseTrainer", "ConformalClassTrainer", "ConformalRegTrainer"]


_TRAINERS = {
    "class": ConformalClassTrainer,
    "reg": ConformalRegTrainer
}

def get_cp_trainer(task, **kwargs):
    if task not in _TRAINERS:
        available = list(_TRAINERS.keys())
        raise ValueError(f"Task '{task}' not found. Available tasks: {available}")
    
    return _TRAINERS[task]
