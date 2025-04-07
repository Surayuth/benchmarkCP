from .base import BaseCP
from .methods.hinge import HingeCP
from .methods.aps import APSCP

# Dictionary-based factory pattern
_METHODS = {
    'hinge': {
        'class': HingeCP,
        'params': [] 
    },
    'aps': {
        'class': APSCP,
        'params': []  
    },
    'raps': {
        'class': None,  # To be implemented
        'params': [
            {'name': 'lamb', 'type': float, 'default': 0.0, 'help': 'Lambda regularization parameter'},
            {'name': 'kreg', 'type': int, 'default': 1, 'help': 'K regularization parameter'}
        ]
    }
    # Add more methods as you implement them
}

def create_cp(method, *args, **kwargs):
    """
    Create a conformal prediction instance.
    
    Args:
        method: String name of the CP method
        *args, **kwargs: Arguments to pass to the CP constructor
        
    Returns:
        An instance of the requested CP class
    """
    if method not in _METHODS:
        available = list(_METHODS.keys())
        raise ValueError(f"Method '{method}' not found. Available methods: {available}")
    
    return _METHODS[method]['class'](*args, **kwargs)
