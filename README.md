# Suitable for Windows - MaskGCT ComfyUI Node Wrapping
## Feature Introduction
1. Amphion-MaskGCT: 0-sample voice synthesis
  [![arXiv](https://img.shields.io/badge/arXiv-Paper-COLOR.svg)](https://arxiv.org/abs/2409.00750)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/amphion/maskgct)
[![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-demo-pink)](https://huggingface.co/spaces/amphion/maskgct)
2. OpenAI-whisper-large-v3: Speech-to-text  [![hf](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-model-yellow)](https://huggingface.co/openai/whisper-large-v3-turbo)
3. Other: Identifying language categories through text / simple audio editing and loading.


## Installation Instructions
1. Install [fmmpeg](https://www.gyan.dev/ffmpeg/builds)
2. If not installed [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases)，
windows download [espeak-ng-X64.msi](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi)，After installation, use the ```espeak-ng --voices``` command to check if the installation was successful (it will return a list of supported languages), without the need to set environment variables.

3. Clone this project using ```git clone ```, or download the zip package and extract it to the ```comfyui/custom_nodes``` directory.

4. Install dependencies: Navigate to the ```.ComfyUI/custom_nodes/ComfyUI_MaskGCT``` directory and open the command line, then type ```(your virtual environment path)/python.exe -m pip install -r requirements.txt``` to install the required packages.

## Update Explanation

Waiting for updates:
1. Audio Speed Adjustment
2. Audio List Editing (Voice List Extraction/Merging/Splitting/Synthesis)
3. Custom Delay Synthesis Based on the End Symbols of Language Segments
4. Change whisper-large-v3-turbo to non networked.

2024/11/07: 
1. Unlimited text length.
2. streamlined node.
3. Change **jieba.cache** from system disk to plugin folder.
4. Update node description
5. Update workflow, The old workflow will no longer be usable

2024/11/06: 
1. First submission

## Nodes Explanation

#### MaskGCT_Load
1. **load_maskgct_model**\
  Load or automatically download maskgct_model

2. **load_w2vbert_model**\
  Load or automatically download w2vbert_model

3. **MaskGCT Pipeline**\
  MaskGCT Preprocessing Pipeline\
  Parameters: \
  -maskgct_model: MaskGCT model.\
  -w2vbert_model: W2V-BERT model.\
  -sample_audio: Audio sample.\
  -sample_prompt_text: Text corresponding to the audio sample.\
  -sample_language: Language of the audio sample.\
  -device: Device to run on.

4. **Load Audio from Path**\
  Load audio files from a specified path.\
  Output:\
  -Audio: Audio data.\
  -Time(s): Time in seconds.\
  -Sample: Sampling rate in Hertz.\
  -Channel: Number of channels.

#### MaskGCT
5. **MaskGCT Run V2**\
  Run MaskGCT to generate speech.\
  Parameters: \
  -maskgct_pipeline: MaskGCT preprocessing pipeline.\
  -target_text: Text for generating speech, with no length limit.\
  -language: The language for generating speech should match the text language (supports 6 languages), it   is recommended to use Auto for automatic recognition.\
  -settings: Optional MaskGCT parameter settings.\
  Output: \
  -Audio: Generated speech.\
  -Audio_List: List of audio after long text cutting.\
  -Batch_Text: List of text after long text cutting.

6. **MaskGCT Setting**\
  MaskGCTParameter settings (default parameters will be used if not connected).\
  model parameter: \
  -maskgct_* prefixed parameters correspond to the configuration of the MaskGCT model.\
  -w2vbert_* prefixed parameters correspond to the configuration of the W2VBert model.\
&emsp;&emsp;w2vbert_time_steps is an optional parameter that is a list containing 12 elements, \
&emsp;&emsp;&emsp;each corresponding to the n_time_steps at 12 different positions.\
&emsp;&emsp;It can be left blank, and will default to ```[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]```.\
&emsp;&emsp;If you want to fine-tune, you can use the ComfyUI-3D-MeshTool plugin to inw2vbert_time_steps.\
  text parameters: \
  -target_len:The length of each output audio segment, 0 indicates automatic calculation. \
&emsp;&emsp;This parameter is invalid when slicing is in effect.\
  -text_slice_length: The length of the text slice, default is 120 English characters, \
&emsp;&emsp;the slice ratio will be automatically adjusted for different languages.\
  -pause_time: The pause time between each sentence (period, exclamation mark, quemark), \
&emsp;&emsp;default is 0.5 seconds.

#### Convert_txt

7. **Speech Recognition-whisper_large_v3**\
  Simple use of the whisper-large-v3-turbo model for speech-to-text.
#### Audio_Edit
8. **Audio Resampling**\
  Adjust the audio sampling rate, whether to resample \
  (not resampling can adjust the audio speed, but the frequency will change).\
  Parameters: \
  -sample: Target sampling rate, range 1 - 4MHz\
  -resample: Whether to resample.

9. **Audio Capture percentage**\
  Trim audio from the beginning and end by percentage.\
  Parameters:\
    -start: Start percentage.\
    -end: End percentage.

10. **Get Audio Data**\
  Retrieve audio data.\
  Output:\
    -sample: Sampling rate.\
    -Time(s): Audio duration in seconds.\
    -channel: Number of channels.\
    -batch_size: Number of audio batches.\
    -Data_Length: Length of audio data.

11. **Get Text Data**\
    Retrieve language text data.\
        Output:\
        -language: Detected language.\
        -character_length: Character length.\
        -words: Array of segmented word strings.\
        -words_number: Number of words.\
        -symbols: Array of segmented symbol strings.\
        -symbols_number: Number of symbols.

12. **Remove Blank Space**\
    Remove silent audio data (for speech-to-text preprocessing).\
        Parameters:\
        -threshold: Audio data with absolute values below this threshold will be removed.\
        -time_length: Limit the processing length to save time.

13. **Multilingual Slice**\
    Divide natural language text into multiple segments according to commas/periods, etc., \
    and merge each segment to a specified length (built-in feature of MaskGCT Run V2).\
    Parameters:\
      -language: Language selection (the reading speed varies for the same paragraph in differenlanguages, \
&emsp;&emsp;and the cutting length will be automatically adjusted through language detection), \
&emsp;&emsp;choose Auto for automatic language recognition.\
      -text_slice_length: Maximum character length, the length of the combined segments will not exceethis value.\
    Output:\
      -sentence: Short sentences divided by punctuation marks (text list).\
      -sentence_number: The number of divided sentences.\
      -length_sentence: Short sentences recombined to the specified length after division (text list).\
      -length_sentence_n: The number of sentences recombined to the specified length after division.

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