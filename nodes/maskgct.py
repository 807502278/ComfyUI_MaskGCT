
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from models.tts.maskgct.maskgct_inference import maskgct_run
# maskgct_run()

import soundfile as sf
import safetensors
import numpy as np

from models.tts.maskgct.maskgct_utils import *
from models.load_model import load_object_dir, load_model_list, get_device_list
from models.load_model import OBJECT_DIR

from transformers import pipeline
import torchaudio
import torchaudio.functional as res
import py3langid

import folder_paths
import logging
import re
from tqdm import tqdm

# logfile_dir = os.path.join(folder_paths.base_path, "comfyui.log")
device_list = get_device_list()


# Translate multi-channel audio data to mono-channel (numpy array),
# support input of ComfyUI standard audio dictionary, tensor, numpy array.
# 多通道音频数据转单通道(np数组)，支持输入ComfyUI标准音频字典，张量，np数组
def audio_to_numpy(audio):
    if isinstance(audio, dict):
        audio = audio["waveform"]
    if isinstance(audio, torch.Tensor):
        if audio.shape[0] > 1:
            audio = audio[0]
            logging.warning(
                "Warning-MaskGCT-Audio_to_Numpy-E01: Audio batch is temporarily not supported, the first audio has been selected")
        audio = audio.squeeze(0)
        audio = np.array(audio.cpu())
    if isinstance(audio, np.ndarray):
        if len(audio.shape) > 1 and audio.shape[0] > 1:
            audio = np.mean(audio, axis=0)
    else:
        logging.error(
            "Error-MaskGCT-Audio_to_Numpy-E02: unsupported audio type", exc_info=True)
    return audio


# Detect language, input text and supported language list/abbreviations, return language abbreviation and whether it is supported.
# 检测语言，输入文本 和支持的语言列表/语言缩写，返回语言缩写和是否支持
def obtain_language(text: str, supported_languages=None):
    language = py3langid.classify(text)[0]
    if supported_languages is None:
        supported = language in ["en", "zh", "ja", "fr", "ko", "de"]
    elif isinstance(supported_languages, str):
        supported = language == supported_languages
    elif isinstance(supported_languages, list):
        supported = language in supported_languages
    else:
        supported = True
    return (language, supported)


# Slice text into fragments, input text and maximum length, return fragment list and combined fragment list.
# 切分文本为片段，输入文本和最大长度，返回片段列表和组合的片段列表
def slice_combination(text, length=100, language=None):
    character_ratio = {"en": 1, "zh": 0.6,
                       "ja": 1.3, "fr": 1.5, "ko": 1, "de": 1.75}
    if language is None:
        language = py3langid.classify(text)[0]
    if language not in character_ratio.keys():
        language = "en"
        logging.warning(
            "Warning-MaskGCT-Slice_Combination: language not supported, use default language 'en' instead")
    length = int(length * character_ratio[language])
    if len(text) <= length:
        return ([text], [text])
    else:
        pattern = r'(?<=[，,。.！!\n])'
        fragments = re.split(pattern, text)
        # remove empty fragments
        fragments = [frag for frag in fragments if frag]
        n = len((fragments))
        if n == 1:
            return ([text], [text])
        elif n == 2:
            return (fragments, fragments)
        else:
            result = []
            combined = ""
            for i in range(n):
                combined += fragments[i]
                if len(combined) >= length:
                    result.append(combined)
                    combined = ""
                elif i == n - 1:
                    result.append(combined)
    return (fragments, result)


CATEGORY_NAME = "MaskGCT/MaskGCT_Load"


class load_maskgct_model:
    DESCRIPTION = """Load or automatically download maskgct_model"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "maskgct_model": (["amphion_maskgct",],),
            },
            "optional": {
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("maskgct_model",)
    RETURN_NAMES = ("maskgct_model",)
    # OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_maskgct_model"

    def load_maskgct_model(self, maskgct_model):
        # build model
        device = device_list[1]
        cfg_path = os.path.join(
            OBJECT_DIR, "models/tts/maskgct/config/maskgct.json")
        cfg = load_config(cfg_path)
        # 1. build semantic model (w2v-bert-2.0)
        # semantic_model, semantic_mean, semantic_std = build_semantic_model(device)
        # 2. build semantic_codec
        semantic_codec = build_semantic_codec(cfg.model.semantic_codec, device)
        # 3. build acoustic codec
        codec_encoder, codec_decoder = build_acoustic_codec(
            cfg.model.acoustic_codec, device
        )
        # 4. build t2s model
        t2s_model = build_t2s_model(cfg.model.t2s_model, device)
        # 5. build s2a model
        s2a_model_1layer = build_s2a_model(
            cfg.model.s2a_model.s2a_1layer, device)
        s2a_model_full = build_s2a_model(cfg.model.s2a_model.s2a_full, device)

        # download checkpoint
        # download semantic codec ckpt
        maskgct_repo_id = "amphion/MaskGCT"
        file_list = [
            "semantic_codec/model.safetensors",
            "acoustic_codec/model.safetensors",
            "acoustic_codec/model_1.safetensors",
            "t2s_model/model.safetensors",
            "s2a_model/s2a_model_1layer/model.safetensors",
            "s2a_model/s2a_model_full/model.safetensors"
        ]
        semantic_code_ckpt, codec_encoder_ckpt, codec_decoder_ckpt, t2s_model_ckpt, s2a_1layer_ckpt, s2a_full_ckpt = load_model_list(
            maskgct_repo_id, file_list=file_list)

        # load semantic codec
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        # load acoustic codec
        safetensors.torch.load_model(codec_encoder, codec_encoder_ckpt)
        safetensors.torch.load_model(codec_decoder, codec_decoder_ckpt)
        # load t2s model
        safetensors.torch.load_model(t2s_model, t2s_model_ckpt)
        # load s2a model
        safetensors.torch.load_model(s2a_model_1layer, s2a_1layer_ckpt)
        safetensors.torch.load_model(s2a_model_full, s2a_full_ckpt)
        model_list = [
            semantic_codec,
            codec_encoder,
            codec_decoder,
            t2s_model,
            s2a_model_1layer,
            s2a_model_full,
        ]
        return (model_list,)


class load_w2vbert_model:
    DESCRIPTION = """Load or automatically download w2vbert_model"""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "w2vbert_model": (["w2v-bert-v2.0",],),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("w2vbert_model",)
    RETURN_NAMES = ("w2vbert_model",)
    # OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_w2vbert_model"

    def load_w2vbert_model(self, w2vbert_model):
        device = device_list[1]
        if w2vbert_model == "w2v-bert-v2.0":
            return (build_semantic_model(device),)
        else:
            logging.error(
                "Error-MaskGCT-load_w2vbert_model: please select a valid model!", exc_info=True)
            return (None,)


class maskgct_pipeline:
    DESCRIPTION = """
        MaskGCT Preprocessing Pipeline
        Parameters: 
        -maskgct_model: MaskGCT model.
        -w2vbert_model: W2V-BERT model.
        -sample_audio: Audio sample.
        -sample_prompt_text: Text corresponding to the audio sample.
        -sample_language: Language of the audio sample.
        -device: Device to run on.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "maskgct_model": ("maskgct_model",),
                "w2vbert_model": ("w2vbert_model",),
                "sample_audio": ("AUDIO",),
                "sample_prompt_text": ("STRING", {"default": "", }),
                "sample_language": (["Auto", "en", "zh", "ja", "fr", "ko", "de",], {"default": "Auto"}),
                "device": (list(device_list[0].keys()), {"default": str(device_list[1])}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("maskgct_pipeline",)
    RETURN_NAMES = ("maskgct_pipeline",)
    FUNCTION = "mc_pipeline"

    def mc_pipeline(self, maskgct_model, w2vbert_model,
                    sample_audio, sample_prompt_text, sample_language,
                    device):
        # Unified equipment 统一设备
        device = device_list[0][device]
        for i in maskgct_model:
            i.to(device)
        for i in w2vbert_model:
            i.to(device)

        # MaskGCT_Pipeline
        pipeline = MaskGCT_Inference_Pipeline(
            w2vbert_model[0],
            *maskgct_model,
            w2vbert_model[1],
            w2vbert_model[2],
            device,
        )

        # Audio sample preprocessing 音频样本预处理
        sample_rate = sample_audio["sample_rate"]
        waveform = audio_to_numpy(sample_audio)
        if sample_rate != 16000:
            waveform_16k = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=16000)
        else:
            waveform_16k = waveform
        if sample_rate != 24000:
            waveform_24k = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=24000)
        else:
            waveform_24k = waveform

        if sample_language == "Auto":
            sample_language = py3langid.classify(sample_prompt_text)[0]
            if not sample_language in ["en", "zh", "ja", "fr", "ko", "de"]:
                logging.error(
                    "Error-MaskGCT-Sample_Audio: language not supported!", exc_info=True)
        return ([pipeline, [waveform_16k, waveform_24k, sample_prompt_text, sample_language]],)


class from_path_load_audio:
    DESCRIPTION = """
    Load audio files from a specified path.
    Output:
    -Audio: Audio data.
    -Time(s): Time in seconds.
    -Sample: Sampling rate in Hertz.
    -Channel: Number of channels.
    """
    @classmethod
    def INPUT_TYPES(s):
        path = os.path.join(folder_paths.base_path,
                            "custom_nodes/ComfyUI_MaskGCT/sample/trump_0.wav")
        return {
            "required": {
                "Audio_FilePath": ("STRING", {"default": path}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("Audio", "Sample", "Time(s)","channel")
    FUNCTION = "load_audio_v2"

    def load_audio_v2(self, Audio_FilePath):
        audio_tensor, sr = torchaudio.load(Audio_FilePath)
        time = audio_tensor.shape[-1]/sr
        audio_tensor = audio_tensor.unsqueeze(0)
        channel = audio_tensor.shape[1]
        return ({"waveform": audio_tensor, "sample_rate": sr}, sr, time,channel)


CATEGORY_NAME = "MaskGCT"


class maskgct_run_v2:
    DESCRIPTION = """
    Run MaskGCT to generate speech.
        Parameters: 
            -maskgct_pipeline: MaskGCT preprocessing pipeline.
            -target_text: Text for generating speech, with no length limit.
            -language: The language for generating speech should match the text language (supports 6 languages), 
                it is recommended to use Auto for automatic recognition.
            -settings: Optional MaskGCT parameter settings.
        Output: 
            -Audio: Generated speech.
            -Audio_List: List of audio after long text cutting.
            -Batch_Text: List of text after long text cutting.
    说明: 基于MaskGCT模型的语音生成。
        输入：
        运行MaskGCT生成语音.
        参数: 
            -maskgct_pipeline: MaskGCT预处理管线。
            -target_text: 用于生成语音的文本，没有长度限制。
            -language: 生成语音的语言应与文本语言对应(支持6种语言),建议使用Auto自动识别。
            -settings: 可选MaskGCT参数设置。
        输出: 
            -Audio: 生成的语音。
            -Audio_List: 长文本切割后的音频列表。
            -Batch_Text: 长文本切割后的文本列表。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "maskgct_pipeline": ("maskgct_pipeline",),
                    "target_text": ("STRING", {"default": "We do not break. We never give in. We never back down.",
                                               "multiline": True}),
                    "language": (["Auto", "en", "zh", "ja", "fr", "ko", "de",], {"default": "Auto"}),
                },
                "optional": {
                    "setting": ("maskgct_setting",),
                }
            }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("Audio", "Audio_List", "Batch_Text")
    FUNCTION = "maskgct_semantic"

    def maskgct_semantic(self,
                         maskgct_pipeline,
                         target_text,
                         language,
                         setting=None,
                         ):

        # Set default values 设置默认值
        if setting == None:
            ms = [25, 2.5, 0.75, [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2.5, 0.75]
            target_len, text_slice_length, pause_time = [0, 120, 0.7]
        else:
            ms = setting[0]
            target_len, text_slice_length, pause_time = setting[1]
        if language == "Auto":
            language = py3langid.classify(target_text)[0]
            if not language in ["en", "zh", "ja", "fr", "ko", "de"]:
                logging.error(
                    "Error-Maskgct_run_v2: language not supported!", exc_info=True)

        # Split Text 切分文本
        if isinstance(target_text, list):
            text_list = target_text
        else:
            _, text_list = slice_combination(
                target_text, text_slice_length, language)
        if target_len == 0 or len(text_list) > 1:
            target_len = None

        # Generate fragmented audio from segmented text 通过切分的文本生成片段音频
        audio_list = []
        audio_data_list = []
        sample_audio = maskgct_pipeline[1]
        for i in tqdm(text_list):
            recovered_audio = maskgct_pipeline[0].maskgct_inference(
                [sample_audio[0], sample_audio[1]],
                sample_audio[2],
                i,
                sample_audio[3],
                language,
                target_len,
                *ms
            )
            recovered_audio = torch.Tensor(recovered_audio, device="cpu")
            recovered_audio = recovered_audio.repeat(1, 1, 1)
            audio_data_list.append(recovered_audio)

        # Prepare for output 准备输出
        if len(audio_data_list) == 1:
            merged_audio = {
                "waveform": audio_data_list[0], "sample_rate": 24000}
            audio_list = recovered_audio
        else:
            pause_time = 0.5
            pause_audio = torch.zeros(1, 1, int(pause_time*24000), device="cpu")
            audio_data_list_pause = []
            for i in range(len(audio_data_list)):
                # Composite audio list 合成音频列表
                audio_list.append(
                    {"waveform": audio_data_list[i], "sample_rate": 24000})
                # Synthesize audio data list and add pauses 合成音频数据列表，并增加停顿
                audio_data_list_pause.append(audio_data_list[i])
                if text_list[i][-1] in [".", "!", "?", "。", "！", "？", "\n"]:
                    audio_data_list_pause.append(pause_audio)
            merged_audio = {"waveform": torch.cat(
                audio_data_list_pause, dim=2), "sample_rate": 24000}
        return (merged_audio, audio_list, text_list)


class maskgct_setting:
    DESCRIPTION = """
        explain:
            1. Model Parameters:
                maskgct_* prefixed parameters correspond to the configuration of the MaskGCT model.
                w2vbert_* prefixed parameters correspond to the configuration of the W2VBert model.
                    w2vbert_time_steps is an optional parameter that is a list containing 12 elements, 
                        each corresponding to the n_time_steps at 12 different positions.
                    It can be left blank, and will default to [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
                    If you want to fine-tune, you can use the ComfyUI-3D-MeshTool plugin to integrate w2vbert_time_steps.
            2. Text Parameters:
                target_len:The length of each output audio segment, 0 indicates automatic calculation. 
                    This parameter is invalid when slicing is in effect.
                text_slice_length: The length of the text slice, default is 120 English characters, 
                    the slice ratio will be automatically adjusted for different languages.
                pause_time: The pause time between each sentence (period, exclamation mark, question mark), 
                    default is 0.5 seconds.

        说明:
            1.模型参数：
                maskgct_* 前缀的参数对应MaskGCT模型的配置。
                w2vbert_* 前缀的参数对应W2VBert模型的配置。
                    w2vbert_time_steps可选参数是一个列表,包含12个元素,分别对应12个位置的n_time_steps
                    可不输入，将默认使用[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                    如果你想精细的调整，可以使用 ComfyUI-3D-MeshTool 插件来接入w2vbert_time_steps
            2.文本参数：
                target_len: 输出的每段音频长度,0表示自动计算。文本切片生效时该参数无效。
                text_slice_length: 文本切片的长度,默认120英文字符,0不切片,不同语言会自动调整切片比例。
                pause_time: 每句话(以句号、感叹号、问号区分)之间的停顿时间,默认0.5秒。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "maskgct_time_steps": ("INT", {"default": 25, "min": 1, "max": 99999}),
                    "maskgct_cfg": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 50.0, "step": 0.01}),
                    "maskgct_rescale_cfg": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 10.0, "step": 0.01}),
                    "w2vbert_cfg": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 50.0, "step": 0.01}),
                    "w2vbert_rescale_cfg": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 10.0, "step": 0.01}),
                    "target_len": ("INT", {"default": 0, "min": 0, "max": 99999}),
                    "text_slice_length": ("INT", {"default": 120, "min": 0, "max": 8192, "step": 1}),
                    "pause_time": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 4.0, "step": 0.01, "display": "slider"})
                },
                "optional": {
                    "w2vbert_time_steps": ("LIST",),
                }
            }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("maskgct_setting",)
    RETURN_NAMES = ("setting",)
    FUNCTION = "setting"

    def setting(self, maskgct_time_steps, maskgct_cfg, maskgct_rescale_cfg, w2vbert_cfg, w2vbert_rescale_cfg,
                target_len, text_slice_length, pause_time,
                w2vbert_time_steps=None):
        if w2vbert_time_steps == None:
            w2vbert_time_steps = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return ([
                [maskgct_time_steps, maskgct_cfg, maskgct_rescale_cfg, w2vbert_time_steps, w2vbert_cfg, w2vbert_rescale_cfg],
                [target_len, text_slice_length, pause_time]
                ],)


CATEGORY_NAME = "MaskGCT/Convert_txt"


class whisper_large_v3:
    DESCRIPTION = """
    Simple use of the whisper-large-v3-turbo model for speech-to-text.
    简单的使用whisper-large-v3-turbo模型进行语音文字。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "chunk_length_s": ("INT", {"default": 30, "min": 1, "max": 99999}),
                "batch_size": ("INT", {"default": 128, "min": 1, "max": 8192}),
                "device": (list(device_list[0].keys()), {"default": str(device_list[1])}),
            },
            "optional": {
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "STRING",)
    RETURN_NAMES = ("audio", "text",)
    FUNCTION = "speech_recognition"

    def speech_recognition(self, audio, chunk_length_s, batch_size, device):
        pipe = pipeline(
            "automatic-speech-recognition",
            model=load_object_dir("openai/whisper-large-v3-turbo"),
            torch_dtype=torch.float16,
            device=device,
        )
        # torch audio to numpy
        audio_numpy = audio_to_numpy(audio)
        print(f"---------------------{audio_numpy.shape}")
        new_audio = {
            "sampling_rate": audio["sample_rate"], "array": audio_numpy}
        prompt_text = pipe(
            new_audio,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        return (audio, prompt_text,)


CATEGORY_NAME = "MaskGCT/Audio_Edit"


class audio_resample:
    DESCRIPTION = """
    Adjust the audio sampling rate, whether to resample 
    (not resampling can adjust the audio speed, but the frequency will change).
        Parameters: 
        -sample: Target sampling rate, range 1 - 4MHz
        -resample: Whether to resample.
    调整音频采样率，是否重采样(不重采样可调整音频速度，但是频率会变)。
        参数: 
        -sample: 目标采样率,范围 1 - 4MHz
        -resample: 是否重采样。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "sample": ("INT", {"default": 24000, "min": 1, "max": 4096000}),
                "re_sample": ("BOOLEAN", {"default": True}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "INT")
    RETURN_NAMES = ("audio", "original_sample")
    FUNCTION = "audio_sampling"

    def audio_sampling(self, audio, sample, re_sample):
        audio_sample = audio["sample_rate"]
        if audio_sample != sample:
            audio_data = audio["waveform"]
            if isinstance(audio_data, torch.Tensor) and re_sample:
                audio_data = res.resample(audio_data, audio_sample, sample)
            elif isinstance(audio_data, np.ndarray) and re_sample:
                audio_data = librosa.resample(
                    audio_data, orig_sr=audio_sample, target_sr=sample)
            else:
                logging.error(
                    "Error-MaskGCT-Audio_resample: unsupported audio data type!", exc_info=True)
            new_audio = {
                "sample_rate": sample,
                "waveform": audio_data
            }
        else:
            new_audio = audio
        return (new_audio, audio_sample)


class audio_capture_percentage:
    DESCRIPTION = """
    Trim audio from the beginning and end by percentage.
    Parameters:
      -start: Start percentage.
      -end: End percentage.\
    从开始和结尾按百分比截取音频。
        参数: 
        -start: 开始百分比。
        -end: 结尾百分比。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "start": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.001, "display": "slider"}),
                "end": ("FLOAT", {"default": 1.0, "min": 0, "max": 1, "step": 0.001, "display": "slider"}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "audio_sampling"

    def audio_sampling(self, audio, start, end):
        audio_data = audio["waveform"]
        new_audio = {}
        if start < end:
            if isinstance(audio_data, torch.Tensor) or isinstance(audio_data, np.ndarray):
                n = audio_data.shape[-1]
                new_audio["waveform"] = audio_data[...,
                                                   int(n*start):int(n*end)]
                new_audio["sample_rate"] = audio["sample_rate"]
            else:
                new_audio = audio
                logging.error(
                    "Error-MaskGCT-audio_capture_percentage-E01: unsupported audio data type!", exc_info=True)
        else:
            new_audio = audio
            logging.error(
                "Error-MaskGCT-audio_capture_percentage-E02: start should be less than end!", exc_info=True)
        return (new_audio,)


class get_audio_data:
    DESCRIPTION = """
    Retrieve audio data.
        Output:
        -sample: Sampling rate.
        -Time(s): Audio duration in seconds.
        -channel: Number of channels.
        -batch_size: Number of audio batches.
        -Data_Length: Length of audio data.

    获取音频数据。
        输出: 
        -sample: 采样率。
        -Time(s): 音频时长(秒)。
        -channel: 通道数。
        -bath_size: 音频批次数量。
        -Data_Length: 音频数据长度。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("sample", "channel", "bath_size", "Time(s)", "Data_Length")
    FUNCTION = "audio_get_data"

    def audio_get_data(self, audio):
        audio_data = audio["waveform"]
        channel = audio_data.shape[1]
        bath_size = audio_data.shape[0]
        n = audio_data.shape[-1]
        time = n/audio["sample_rate"]
        return (audio["sample_rate"],channel, bath_size,time, n)


class get_text_data:
    DESCRIPTION = """
    Retrieve language text data.
        Output:
        -language: Detected language.
        -character_length: Character length.
        -words: Array of segmented word strings.
        -words_number: Number of words.
        -symbols: Array of segmented symbol strings.
        -symbols_number: Number of symbols.
    获取语言文本数据。
        输出: 
        -language: 识别出的语言。
        -character_length: 字符长度。
        -words: 分割的单词字符串数组。
        -words_number: 单词数量。
        -symbols: 分割的符号字符串数组。
        -symbols_number: 符号数量。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "We do not break. We never give in. We never back down."}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "INT",)
    RETURN_NAMES = ("language", "character_length", "words", "word_number",
                    "symbols", "symbol_number")
    FUNCTION = "get_language"

    def get_language(self, text):
        language, _ = obtain_language(text)
        words = re.findall(r'\b\w+\b', text)
        word_number = len(words)
        symbols = re.findall(r'[^\w\s]', text)
        symbol_number = len(symbols)
        return (language, len(text), words, word_number, symbols, symbol_number)


class remove_blank_space:
    DESCRIPTION = """
    Remove silent audio data (for speech-to-text preprocessing).
        Parameters:
        -threshold: Audio data with absolute values below this threshold will be removed.
        -time_length: Limit the processing length to save time.

    去除静音音频数据(用于音频转文字预处理)。\
        参数: \
        -threshold: 低于该阈值的绝对值音频数据都将被移除\
        -time_length: 限制处理长度，可节省时间。\
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Audio": ("AUDIO",),
                "threshold": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001}),
                "time_length": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 99999.0, "step": 0.01})
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("Audio",)
    FUNCTION = "remove_blank_space"

    def remove_blank_space(self, Audio, threshold, time_length):
        audio_data = Audio["waveform"]
        time_length = time_length*Audio["sample_rate"]
        new_audio_data = self.remove_audio_blanks(
            audio_data, threshold, time_length)
        new_audio = {}
        new_audio["waveform"] = new_audio_data
        new_audio["sample_rate"] = Audio["sample_rate"]
        return (new_audio,)

    def remove_audio_blanks(self, audio_tensor, threshold, max_length):
        """
        Remove blank segments from audio data and merge into a new audio tensor.
        Parameters:
        audio_tensor (torch.Tensor): An audio tensor with shape (channels, samples).
        threshold (float): The threshold used to determine blank segments.
        max_length (int): The maximum length of the merged audio tensor.
        Returns:
        torch.Tensor: The processed audio tensor.

        移除音频数据中的空白片段并合并成新的音频张量。
        参数:
        audio_tensor (torch.Tensor): 形状为(通道, 样本)的音频张量。
        threshold (float): 用于确定空白片段的阈值。
        max_length (int): 合并后的音频张量的最大长度。
        返回:
        torch.Tensor: 处理后的音频张量。
        """
        original_n = False
        if len(audio_tensor.shape) == 3:
            original_n = True
            audio_tensor = audio_tensor.squeeze(0)
        if len(audio_tensor.shape) == 2:
            # Calculate the absolute value of audio for detecting blank segments 计算音频的绝对值，用于检测空白片段
            abs_audio = torch.abs(audio_tensor)
            # Find the start and end indexes of non blank fragments 找到非空白片段的起始和结束索引
            non_silence_intervals = []
            current_start = 0
            accumulate = 0
            for i in range(1, abs_audio.shape[1]):
                # Detecting transition points between blank and non blank segments 检测空白片段和非空白片段的转换点
                if torch.max(abs_audio[:, i-1]) < threshold and torch.max(abs_audio[:, i]) >= threshold:
                    current_start = i
                elif torch.max(abs_audio[:, i-1]) >= threshold and torch.max(abs_audio[:, i]) < threshold:
                    # Calculate the length of the current segment 计算当前片段的长度
                    segment_length = i - current_start
                    # If the current fragment is added and does not exceed the maximum length, it will be added to the list
                    # 如果加上当前片段后不超过max_length，则添加到列表中
                    if accumulate + segment_length <= max_length:
                        non_silence_intervals.append((current_start, i-1))
                        accumulate += segment_length
                    else:
                        break
            # Merge Fragments
            new_audio = torch.cat([audio_tensor[:, start:end+1]
                                  for start, end in non_silence_intervals], dim=1)
            if original_n:
                new_audio = new_audio.repeat(1, 1, 1)
        else:
            new_audio = audio_tensor
            logging.warning(
                "Warningror-MaskGCT-remove_blank_space: The input audio tensor must have 2 or 3 dimensions.")
        return new_audio


class multilingual_slice:
    DESCRIPTION = """
    Divide natural language text into multiple segments according to commas/periods, etc., 
    and merge each segment to a specified length (built-in feature of MaskGCT Run V2).
        Parameters:
          -language: Language selection (the reading speed varies for the same paragraph in different languages, 
                and the cutting length will be automatically adjusted through language detection), 
                choose Auto for automatic language recognition.
          -text_slice_length: Maximum character length, the length of the combined segments will not exceed this value.
        Output:
          -sentence: Short sentences divided by punctuation marks (text list).
          -sentence_number: The number of divided sentences.
          -length_sentence: Short sentences recombined to the specified length after division (text list).
          -length_sentence_n: The number of sentences recombined to the specified length after division.

    将自然语言文本按照逗号/句号等分成多个片段，并将每个片段合并到指定长度(MaskGCT Run V2内置功能)。
        参数：
        -language: 语言选择(同一段话不同语言的阅读速度不同，将通过语言自动调整切割长度), 选Auto自动识别语言。
        -text_slice_length: 最大字符长度，片段组合的长度不会超过此值。
        输出：
        -sentence: 按标点符号分割后的短句(文本列表)。
        -sentence_number: 分割后的句子数量。
        -length_sentence: 分割后重新组合到指定长度后的短句(文本列表)。
        -length_sentence_n: 分割后重新组合到指定长度后的句子数量。
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "We do not break. We never give in. We never back down."}),
                "language": (["Auto", "en", "zh", "ja", "fr", "ko", "de",], {"default": "Auto"}),
                "text_slice_length": ("INT", {"default": 100, "min": 0, "max": 8192}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING","INT","STRING","INT")
    RETURN_NAMES = ("sentence","sentence_number","length_sentence","length_sentence_n",)
    FUNCTION = "multilingual_slice"

    def multilingual_slice(self, text, language, text_slice_length):
        if language == "Auto":
            language = None
        sentence, length_sentence = slice_combination(text, text_slice_length, language)
        return (sentence,len(sentence),length_sentence,len(text))


NODE_CLASS_MAPPINGS = {
    # MaskGCT/MaskGCT_Load
    "load_maskgct_model": load_maskgct_model,
    "load_w2vbert_model": load_w2vbert_model,
    "maskgct_pipeline": maskgct_pipeline,
    "from_path_load_audio": from_path_load_audio,
    # MaskGCT
    "maskgct_run_v2": maskgct_run_v2,
    "maskgct_setting": maskgct_setting,
    # MaskGCT/Convert_txt
    "whisper_large_v3": whisper_large_v3,
    # MaskGCT/Audio_Edit
    "audio_resample": audio_resample,
    "audio_capture_percentage": audio_capture_percentage,
    "get_audio_data": get_audio_data,
    "get_text_data": get_text_data,
    "remove_blank_space": remove_blank_space,
    "multilingual_slice": multilingual_slice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # MaskGCT/MaskGCT_Load
    "load_maskgct_model": "Load MaskGCT Model",
    "load_w2vbert_model": "Load W2VBert Model",
    "maskgct_pipeline": "MaskGCT Pipeline",
    "from_path_load_audio": "Load Audio from Path",
    # MaskGCT
    "maskgct_run_v2": "MaskGCT Run V2",
    "maskgct_setting": "MaskGCT Setting",
    # MaskGCT/Convert_txt
    "whisper_large_v3": "Speech Recognition-whisper_large_v3",
    # MaskGCT/Audio_Edit
    "audio_resample": "Audio Resampling",
    "audio_capture_percentage": "Audio Capture percentage",
    "get_audio_data": "Get Audio Data",
    "get_text_data": "Get Text Data",
    "remove_blank_space": "Remove Blank Space",
    "multilingual_slice": "Multilingual Slice",
}
