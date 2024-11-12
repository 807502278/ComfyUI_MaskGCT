import torch
import numpy as np
import logging
import py3langid
import re

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
        audio = audio.squeeze()
        audio = np.array(audio.cpu())
    if isinstance(audio, np.ndarray):
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)
            audio = audio.squeeze()
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
