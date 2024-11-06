# @ComfyUI-Impact-Pack https://github.com/ltdrdata/ComfyUI-Impact-Pack
import glob
import importlib.util
import os
import sys

load_path = os.path.dirname(__file__)
if load_path not in sys.path:
    sys.path.insert(0, load_path)

extension_folder = os.path.dirname(os.path.realpath(__file__))
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
pyPath = os.path.join(extension_folder,'nodes')

def loadCustomNodes():
    find_files = glob.glob(os.path.join(pyPath, "*.py"), recursive=True)
    for file in find_files:
        file_relative_path = file[len(extension_folder):]
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

loadCustomNodes()# load Custom Nodes

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

