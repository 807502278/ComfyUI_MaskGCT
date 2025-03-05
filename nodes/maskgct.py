
# Copyright (c) 2024 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# from models.tts.maskgct.maskgct_inference import maskgct_run
# maskgct_run()

import soundfile as sf
import safetensors
import numpy as np

from ..models.tts.maskgct.maskgct_utils import *
from ..models.load_model import load_model_list, get_device_list, OBJECT_DIR
from ..models.auto_tool import audio_to_numpy, slice_combination

import torchaudio
import py3langid

import folder_paths
import logging
import re
from tqdm import tqdm

# logfile_dir = os.path.join(folder_paths.base_path, "comfyui.log")
device_list = get_device_list()

CATEGORY_NAME = "MaskGCT/MaskGCT_LoadModel"


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
        # semantic_code_ckpt, codec_encoder_ckpt, codec_decoder_ckpt, 
        # t2s_model_ckpt, s2a_1layer_ckpt, s2a_full_ckpt, file_path
        sc, ce, cd, t2s, s2a, s2a_f, file_path = load_model_list(
            maskgct_repo_id, file_list=file_list)

        # load semantic codec
        safetensors.torch.load_model(semantic_codec, sc)
        # load acoustic codec
        safetensors.torch.load_model(codec_encoder, ce)
        safetensors.torch.load_model(codec_decoder, cd)
        # load t2s model
        safetensors.torch.load_model(t2s_model, t2s)
        # load s2a model
        safetensors.torch.load_model(s2a_model_1layer, s2a)
        safetensors.torch.load_model(s2a_model_full, s2a_f)
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
        for i in w2vbert_model[:-1]:
            i.to(device)

        # MaskGCT_Pipeline
        # maskgct_model: [semantic_codec, codec_encoder, codec_decoder, 
        #                 t2s_model, s2a_model_1layer, s2a_model_full, 
        #                 file_path]
        # w2vbert_model: [semantic_model, semantic_mean, semantic_std, w2v_bert]
        # MaskGCT_Inference_Pipeline:
        #                 semantic_codec, 
        #                 codec_encoder,
        #                 codec_decoder,
        #                 t2s_model,
        #                 s2a_model_1layer,
        #                 s2a_model_full,
        #                 semantic_model, # w2vbert_model
        #                 semantic_mean,  # w2vbert_model
        #                 semantic_std,   # w2vbert_model
        #                 w2v_bert,       # w2vbert_model
        #                 device,
        pipeline = MaskGCT_Inference_Pipeline(
            *maskgct_model,
            *w2vbert_model,
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
    RETURN_NAMES = ("Audio", "Sample", "Time(s)", "channel")
    FUNCTION = "load_audio_v2"

    def load_audio_v2(self, Audio_FilePath):
        audio_tensor, sr = torchaudio.load(Audio_FilePath)
        time = audio_tensor.shape[-1]/sr
        audio_tensor = audio_tensor.unsqueeze(0)
        channel = audio_tensor.shape[1]
        return ({"waveform": audio_tensor, "sample_rate": sr}, sr, time, channel)


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
            pause_audio = torch.zeros(
                1, 1, int(pause_time*24000), device="cpu")
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
                [maskgct_time_steps, maskgct_cfg, maskgct_rescale_cfg,
                    w2vbert_time_steps, w2vbert_cfg, w2vbert_rescale_cfg],
                [target_len, text_slice_length, pause_time]
                ],)


NODE_CLASS_MAPPINGS = {
    # MaskGCT/MaskGCT_LoadModels
    "load_maskgct_model": load_maskgct_model,
    "load_w2vbert_model": load_w2vbert_model,
    "maskgct_pipeline": maskgct_pipeline,
    "from_path_load_audio": from_path_load_audio,
    # MaskGCT
    "maskgct_run_v2": maskgct_run_v2,
    "maskgct_setting": maskgct_setting,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    # MaskGCT/MaskGCT_LoadModels
    "load_maskgct_model": "Load MaskGCT Model",
    "load_w2vbert_model": "Load W2VBert Model",
    "maskgct_pipeline": "MaskGCT Pipeline",
    "from_path_load_audio": "Load Audio from Path",
    # MaskGCT
    "maskgct_run_v2": "MaskGCT Run V2",
    "maskgct_setting": "MaskGCT Setting",
}
