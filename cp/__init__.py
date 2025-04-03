import importlib

def get_cp_instance(cp_method, *args, **kwargs):
    module = importlib.import_module(f"cp.{cp_method}")
    CPClass = getattr(module, "CP")
    return CPClass(*args, **kwargs)