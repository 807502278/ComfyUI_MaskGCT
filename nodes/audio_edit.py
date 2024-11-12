import torch
import numpy as np
import logging
import torchaudio.functional as res
import librosa
import re

from models.auto_tool import obtain_language, slice_combination


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


class audio_scale:
    DESCRIPTION = """
    Audio time duration adjustment
        Parameter: - scale: Audio time duration multiplier, range 0.001-99999
    音频速度调整
        参数: -scale: 音频时长倍数,范围 0.001-99999
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "scale": ("FLOAT", {"default": 1, "min": 0.001, "max": 99999, "step": 0.001}),
            }
        }
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "audio_scale"

    def audio_scale(self, audio, scale):
        if scale == 1:
            return (audio,)
        else:
            new_audio = {}
            new_audio["sample_rate"] = audio["sample_rate"]
            audio_data = torch.tensor([])
            audio_data = audio["waveform"]
            s=list(audio_data.shape)
            s[1] = 0;s[2] = int(round(s[2]/scale))
            audio_data_new = torch.zeros(*s)
            for i in audio_data[0]:
                new_i = np.array(i)
                new_i = librosa.effects.time_stretch(new_i, rate = scale)
                new_i = torch.tensor(new_i)
                audio_data_new = torch.cat((audio_data_new, new_i.repeat(1,1,1)), dim=1)
            new_audio["waveform"] = audio_data_new
        return (new_audio,)


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
        return (audio["sample_rate"], channel, bath_size, time, n)


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
    RETURN_TYPES = ("STRING", "INT", "STRING", "INT")
    RETURN_NAMES = ("sentence", "sentence_number",
                    "length_sentence", "length_sentence_n",)
    FUNCTION = "multilingual_slice"

    def multilingual_slice(self, text, language, text_slice_length):
        if language == "Auto":
            language = None
        sentence, length_sentence = slice_combination(
            text, text_slice_length, language)
        return (sentence, len(sentence), length_sentence, len(text))


NODE_CLASS_MAPPINGS = {
    "audio_resample": audio_resample,
    "audio_scale": audio_scale,
    "audio_capture_percentage": audio_capture_percentage,
    "get_audio_data": get_audio_data,
    "get_text_data": get_text_data,
    "remove_blank_space": remove_blank_space,
    "multilingual_slice": multilingual_slice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "audio_resample": "Audio Resampling",
    "audio_scale": "Audio Speed Adjustment",
    "audio_capture_percentage": "Audio Capture percentage",
    "get_audio_data": "Get Audio Data",
    "get_text_data": "Get Text Data",
    "remove_blank_space": "Remove Blank Space",
    "multilingual_slice": "Multilingual Slice",
}
