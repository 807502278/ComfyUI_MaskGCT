
import os
from huggingface_hub import hf_hub_download, snapshot_download
import torch


def dir_up_level(dir, level=1):
    """
    return the directory with the specified number of levels up.
    获取指定层级上级目录
    """
    # dir=os.sep.join(dir.split(os.sep)[:-1])
    dir = dir.split(os.sep)
    if level >= len(dir):
        dir = [0]
    else:
        dir = os.sep.join(dir[:-1*level])
    return dir


_path = os.path.abspath(__file__)
OBJECT_DIR = dir_up_level(_path, 2)
ESPEAK_DLL = os.path.join(OBJECT_DIR, 'dll/libespeak-ng.dll')  # espeak-ng dll file
ESPEAK_DIR = os.path.join(OBJECT_DIR, 'dll/') 
MODELS_DIR = os.path.join(dir_up_level(_path, 4), "models/maskgct")

MODELS_JIEBA= os.path.join(OBJECT_DIR,'models/jieba')

def load_model_list(repo_id, file_list: list, local_dir=MODELS_DIR, revision: str = None, object_dir=False):
    """
     Download the file list of the specified Hugging Face project to a local directory (preserving the original directory structure, 
        and .cache will also be directed to that directory), return a list of paths or the project path.

    将指定抱脸项目的文件列表下载到本地目录(会保留原目录结构.cache也会指定到该目录)，返回路径列表或项目路径
    """
    local_dir = os.path.join(local_dir, repo_id.replace("\\", "/"))
    cache_dir = os.path.join(dir_up_level(local_dir), "cache")
    dir_list = []
    for name in file_list:
        path_name, model_name = os.path.split(name)
        # model_dir = os.path.join(local_dir, path_name)
        print(f"inspect {path_name}...")
        dir_list.append(
            hf_hub_download(
                repo_id,
                filename=name,
                cache_dir=cache_dir,
                local_dir=local_dir,
                revision=revision
            )
        )
    if object_dir:
        return local_dir
    else:
        return dir_list


def load_object_dir(repo_id, ignore_patterns: list = None, local_dir=MODELS_DIR, revision: str = None):
    """
     Download the Hugging Face project to a local directory (a new directory with the same name as repo_id will be created, 
        and .cache will also be directed to that directory), return the project path.

    将抱脸项目下载到本地目录(会新建repo_id同名目录并将.cache也指定到该目录)，返回项目路径
    """
    local_dir = os.path.join(local_dir, repo_id.replace("\\", "/"))
    cache_dir = os.path.join(dir_up_level(local_dir), "cache")
    object_dir = snapshot_download(repo_id,
                                   ignore_patterns=ignore_patterns,
                                   cache_dir=cache_dir,
                                   local_dir=local_dir)
    return object_dir


def get_device_list():
    device_str = ["default", "cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_str.append(f"cuda:{i}")
    if torch.backends.mps.is_available():
        device_str.append("mps:0")
    n = len(device_str)
    if n > 2:  # default device 默认设备
        device_default = torch.device(device_str[2])
    else:
        device_default = torch.device(device_str[1])

    # Establish a device list dictionary 建立设备列表字典
    device_list = {device_str[0]: device_default, }
    for i in range(n-1):
        device_list[device_str[i+1]] = torch.device(device_str[i+1])

    return [device_list, device_default]

# Download to the predetermined path
# https://huggingface.co/docs/huggingface_hub/guides/download
# Document Query
