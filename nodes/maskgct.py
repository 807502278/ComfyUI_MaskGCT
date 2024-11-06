
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


CATEGORY_NAME = "MaskGCT/MaskGCT_Load"


class load_maskgct_model:
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "maskgct_model": ("maskgct_model",),
                "w2vbert_model": ("w2vbert_model",),
                "device": (list(device_list[0].keys()), {"default": str(device_list[1])}),
            },
            "optional": {
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("maskgct_pipeline",)
    RETURN_NAMES = ("maskgct_pipeline",)
    FUNCTION = "pipeline"

    def pipeline(self, maskgct_model, w2vbert_model, device):
        device = device_list[0][device]
        for i in maskgct_model:
            i.to(device)
        for i in w2vbert_model:
            i.to(device)
        pipeline = MaskGCT_Inference_Pipeline(
            w2vbert_model[0],
            *maskgct_model,
            w2vbert_model[1],
            w2vbert_model[2],
            device,
        )
        return (pipeline,)


class sample_audio:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "prompt_text": ("STRING", {"default": "", }),
                "language": (["Auto", "en", "zh", "ja", "fr", "ko", "de",], {"default": "Auto"}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("audio_pipe",)
    RETURN_NAMES = ("sample_audio",)
    FUNCTION = "audio_sample"

    def audio_sample(self, audio, prompt_text, language):
        sample_rate = audio["sample_rate"]
        waveform = audio_to_numpy(audio)
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

        if language == "Auto":
            language = py3langid.classify(prompt_text)[0]
            if not language in ["en", "zh", "ja", "fr", "ko", "de"]:
                logging.error(
                    "Error-MaskGCT-Sample_Audio: language not supported!", exc_info=True)
        return ([waveform_16k, waveform_24k, prompt_text, language],)


class from_path_load_audio:
    @classmethod
    def INPUT_TYPES(s):
        path=os.path.join(folder_paths.base_path, "custom_nodes/ComfyUI_MaskGCT/sample/trump_0.wav")
        return {
            "required": {
                "Audio_FilePath": ("STRING",{"default": path}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO", "INT", "FLOAT")
    RETURN_NAMES = ("Audio", "Sample", "Time(s)")
    FUNCTION = "load_audio_v2"

    def load_audio_v2(self, Audio_FilePath):
        audio_tensor, sr = torchaudio.load(Audio_FilePath)
        time = audio_tensor.shape[-1]/sr
        audio_tensor = audio_tensor.unsqueeze(0)
        return ({"waveform": audio_tensor, "sample_rate": sr}, sr, time,)


CATEGORY_NAME = "MaskGCT"


class maskgct_run_v2:
    DESCRIPTION = """
    When target_len is 0, the length is automatically calculated.
    Note: If target_len exceeds the length corresponding to normal speaking speed, 
        the first half of the audio may become garbled.

    target_len长度为 0 时，自动计算长度。
    注意：target_len超过正常语速时的长度时，音频前半部分会出现乱码。
    """

    @classmethod
    def INPUT_TYPES(s):
        wav_path = os.path.join(OBJECT_DIR, "sample/trump_0.wav")
        return {
            "required": {
                "maskgct_pipeline": ("maskgct_pipeline",),
                "sample_audio": ("audio_pipe",),
                "target_text": ("STRING", {"default": "We do not break. We never give in. We never back down.",
                                           "multiline": True}),
                "language": (["Auto", "en", "zh", "ja", "fr", "ko", "de",], {"default": "Auto"}),
                "target_len": ("INT", {"default": 0, "min": 0, "max": 99999}),
            },
            "optional": {
                "maskgct_setting": ("maskgct_setting",),
                "w2vbert_setting": ("w2vbert_setting",)
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("AUDIO",)
    FUNCTION = "maskgct_semantic"

    def maskgct_semantic(self,
                         maskgct_pipeline,
                         sample_audio,
                         target_text,
                         language,
                         target_len,
                         maskgct_setting=None,
                         w2vbert_setting=None,
                         ):
        # speech_16k = librosa.load(sample_wav_path, sr=16000)[0]
        if target_len == 0:
            target_len = None
        if language == "Auto":
            language = py3langid.classify(target_text)[0]
            if not language in ["en", "zh", "ja", "fr", "ko", "de"]:
                logging.error(
                    "Error-Maskgct_run_v2: language not supported!", exc_info=True)
        if maskgct_setting == None:
            maskgct_setting = [25, 2.5, 0.75]
        if w2vbert_setting == None:
            w2vbert_setting = [
                [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 2.5, 0.75]

        recovered_audio = maskgct_pipeline.maskgct_inference(
            [sample_audio[0], sample_audio[1]],
            sample_audio[2],
            target_text,
            sample_audio[3],
            language,
            target_len,
            *maskgct_setting,
            *w2vbert_setting
        )
        recovered_audio = torch.Tensor(recovered_audio, device="cpu")
        recovered_audio = recovered_audio.repeat(1, 1, 1)
        return ({"waveform": recovered_audio, "sample_rate": 24000},)


class maskgct_setting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "n_time_steps": ("INT", {"default": 25, "min": 1, "max": 99999}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 50.0, "step": 0.01}),
                "rescale_cfg": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("maskgct_setting",)
    RETURN_NAMES = ("maskgct_setting",)
    FUNCTION = "setting"

    def setting(self, n_time_steps, cfg, rescale_cfg):
        return ([n_time_steps, cfg, rescale_cfg],)


class w2vbert_setting:
    DESCRIPTION = """
        explain:
        n_time_steps is a list that contains 12 elements, 
            corresponding to the n_time_steps at 12 different positions.
        You do not need to input it; by default, it uses [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].
        If you wish to make fine adjustments,
            You can also use my ComfyUI-3D-MeshTool plugin to generate the n_time_steps for w2vbert.
        说明:
        n_time_steps是一个列表，包含12个元素，分别对应12个位置的n_time_steps。
        可以不输入，默认使用[25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        如果你想精细的调整，可以使用 ComfyUI-3D-MeshTool 插件来生成w2vbert的n_time_steps后输入
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.01, "max": 50.0, "step": 0.01}),
                "rescale_cfg": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "n_time_steps": ("LIST",),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("w2vbert_setting",)
    RETURN_NAMES = ("w2vbert_setting",)
    FUNCTION = "setting"

    def setting(self, cfg, rescale_cfg, n_time_steps=None):
        if n_time_steps == None:
            n_time_steps = [25, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        return ([n_time_steps, cfg, rescale_cfg],)


CATEGORY_NAME = "MaskGCT/Convert_txt"


class whisper_large_v3:
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
        if audio_sample != sample :
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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("INT", "FLOAT", "INT")
    RETURN_NAMES = ("sample", "Time(s)", "Length(samples)")
    FUNCTION = "audio_get_data"

    def audio_get_data(self, audio):
        audio_data = audio["waveform"]
        n = audio_data.shape[-1]
        time = n/audio["sample_rate"]
        return (audio["sample_rate"], time, n)


class get_text_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "We do not break. We never give in. We never back down."}),
            },
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "INT",)
    RETURN_NAMES = ("language", "words", "word_number",
                    "symbols", "symbol_number")
    FUNCTION = "get_language"

    def get_language(self, text):
        language, _ = obtain_language(text)
        characters_length = len(text)
        words = re.findall(r'\b\w+\b', text)
        word_number = len(words)
        symbols = re.findall(r'[^\w\s]', text)
        symbol_number = len(symbols)
        return (language, words, word_number, symbols, symbol_number)


class remove_blank_space:
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


NODE_CLASS_MAPPINGS = {
    # MaskGCT/MaskGCT_Load
    "load_maskgct_model": load_maskgct_model,
    "load_w2vbert_model": load_w2vbert_model,
    "maskgct_pipeline": maskgct_pipeline,
    "sample_audio": sample_audio,
    "from_path_load_audio": from_path_load_audio,
    # MaskGCT
    "maskgct_run_v2": maskgct_run_v2,
    "maskgct_setting": maskgct_setting,
    "w2vbert_setting": w2vbert_setting,
    # MaskGCT/Convert_txt
    "whisper_large_v3": whisper_large_v3,
    # MaskGCT/Audio_Edit
    "audio_resample": audio_resample,
    "audio_capture_percentage": audio_capture_percentage,
    "get_audio_data": get_audio_data,
    "get_text_data": get_text_data,
    "remove_blank_space": remove_blank_space,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # MaskGCT/MaskGCT_Load
    "load_maskgct_model": "Load MaskGCT Model",
    "load_w2vbert_model": "Load W2VBert Model",
    "maskgct_pipeline": "MaskGCT Pipeline",
    "sample_audio": "Sample Audio",
    "from_path_load_audio": "Load Audio from Path",
    # MaskGCT
    "maskgct_run_v2": "MaskGCT Run V2",
    "maskgct_setting": "MaskGCT Setting",
    "w2vbert_setting": "W2VBert Setting",
    # MaskGCT/Convert_txt
    "whisper_large_v3": "Speech Recognition-whisper_large_v3",
    # MaskGCT/Audio_Edit
    "audio_resample": "Audio Resampling",
    "audio_capture_percentage": "Audio Capture percentage",
    "get_audio_data": "Get Audio Data",
    "get_text_data": "Get Text Data",
    "remove_blank_space": "Remove Blank Space",
}
