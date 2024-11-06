# Suitable for Windows - MaskGCT ComfyUI Node Wrapping
## Feature Introduction
1. Amphion-MaskGCT：0-sample voice synthesis
  [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct)
2. OpenAI-whisper-large-v3：Speech-to-text  [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/openai/whisper-large-v3-turbo)
3. Other：Identifying language categories through text / simple audio editing and loading.


## Installation Instructions
1. Install [fmmpeg](https://www.gyan.dev/ffmpeg/builds)
2. If not installed [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)，
windows download [espeak-ng-X64.msi](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi)，After installation, use the ```espeak-ng --voices``` command to check if the installation was successful (it will return a list of supported languages), without the need to set environment variables.

3. Clone this project using ```git clone ```, or download the zip package and extract it to the ```comfyui/custom_nodes``` directory.

4. Install dependencies: Navigate to the ```.ComfyUI/custom_nodes/ComfyUI_MaskGCT``` directory and open the command line, then type ```(your virtual environment path)/python.exe -m pip install -r requirements.txt``` to install the required packages.



## Model Download
The model will automatically be downloaded from the source to the comfyui/models/maskgct directory.

If you need to download the model manually, please organize the files as per the following structure.

Note: When downloading individual files manually, the name of config.json may change.
Download link (If you keep the original project folder structure during download, then you don't have to compare the files.
):

[whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo/tree/main)

[w2v-bert-2.0](https://huggingface.co/facebook/w2v-bert-2.0/tree/main)

[amphion-large-v3-turbo](https://huggingface.co/amphion/MaskGCT/tree/main)

```
models
    └── maskgct
        ├── openai
        │   └── whisper-large-v3-turbo
        │       └── (14 files ... )
        ├── facebook
        │   └── w2v-bert-2.0
        │       ├── config.json
        │       ├── model.safetensors
        │       └── preprocessor_config.json
        └── amphion
            └── amphion-large-v3-turbo
                └── MaskGCT
                    ├── acoustic_codec
                    │   ├── model.safetensors
                    │   └── model_1.safetensors
                    ├── s2a_model
                    │   ├── s2a_model_1layer
                    │   │   └── model.safetensors
                    │   └── s2a_model_full
                    │       └── model.safetensors
                    ├── semantic_codec
                    │   └── model.safetensors
                    └── t2s_model
                        └── model.safetensors
```

## Known Issues
### Errors
1. espeak-ng is not installed.

```RuntimeError: espeak not installed on your system```
or
```...\poly_bert_model.onnx...cannot find file```

2. **pyopenjtalk** Dependency installation failed:

Check if the following path is in the environment variable PATH (my VC version here is 14.35.32215).

D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.35.32215\bin\Hostx86\x64

D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin



### Ignorable warnings.

1. 
```
UserWarning: Plan failed with a cudnnException: CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR: cudnnFinalize Descriptor Failed cudnn_status: CUDNN_STATUS_NOT_SUPPORTED
```
Switch to torch>=2.3.1 to resolve (not recommended).

2. 
```
UserWarning: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
```

## Citations

If you use MaskGCT in your research, please cite the following paper:

```bibtex
@article{wang2024maskgct,
  title={MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer},
  author={Wang, Yuancheng and Zhan, Haoyue and Liu, Liwei and Zeng, Ruihong and Guo, Haotian and Zheng, Jiachen and Zhang, Qiang and Zhang, Xueyao and Zhang, Shunsi and Wu, Zhizheng},
  journal={arXiv preprint arXiv:2409.00750},
  year={2024}
}

@inproceedings{amphion,
    author={Zhang, Xueyao and Xue, Liumeng and Gu, Yicheng and Wang, Yuancheng and Li, Jiaqi and He, Haorui and Wang, Chaoren and Song, Ting and Chen, Xi and Fang, Zihao and Chen, Haopeng and Zhang, Junan and Tang, Tze Ying and Zou, Lexiao and Wang, Mingxuan and Han, Jun and Chen, Kai and Li, Haizhou and Wu, Zhizheng},
    title={Amphion: An Open-Source Audio, Music and Speech Generation Toolkit},
    booktitle={{IEEE} Spoken Language Technology Workshop, {SLT} 2024},
    year={2024}
}
```


BibTeX entry and citation info
```bibtex
@misc{radford2022whisper,
  doi = {10.48550/ARXIV.2212.04356},
  url = {https://arxiv.org/abs/2212.04356},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya},
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```