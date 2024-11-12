# @ComfyUI-Impact-Pack https://github.com/ltdrdata/ComfyUI-Impact-Pack
import glob
import importlib.util
import os
import sys

self_path = os.path.dirname(os.path.realpath(__file__))
#if load_path not in sys.path: sys.path.insert(0, self_path)

class TempSysPath:
    def __init__(self, path_to_add):
        self.path_to_add = path_to_add
        self.original_path = sys.path.copy()

    def __enter__(self):
        sys.path.insert(0, self.path_to_add)

    def __exit__(self, type, value, traceback):
        sys.path = self.original_path

with TempSysPath(self_path):
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    pyPath = os.path.join(self_path,'nodes')
    find_files = glob.glob(os.path.join(pyPath, "*.py"), recursive=True)
    
    for file in find_files:
        file_relative_path = file[len(self_path):]
        model_name = file_relative_path.replace(os.sep, '.')
        model_name = os.path.splitext(model_name)[0]
        module = importlib.import_module(model_name, __name__)
        # if NODE_CLASS_MAPPINGS not none, update it
        if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
            NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
            # if NODE_DISPLAY_NAME_MAPPINGS not none, update it
            if hasattr(module, "NODE_DISPLAY_NAME_MAPPINGS") and getattr(module, "NODE_DISPLAY_NAME_MAPPINGS") is not None:
                NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)
        if hasattr(module, "init"):# init
            getattr(module, "init")()
    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

