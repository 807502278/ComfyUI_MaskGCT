import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset

from models.auto_tool import audio_to_numpy
from models.load_model import load_object_dir, get_device_list
device_list = get_device_list()

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
    RETURN_TYPES = ("AUDIO","STRING",)
    RETURN_NAMES = ("audio","text",)
    FUNCTION = "speech_recognition"

    def speech_recognition(self, audio, chunk_length_s, batch_size, device):
        TRANSFORMERS_OFFLINE=1
        pipe = pipeline(
            "automatic-speech-recognition",
            model=load_object_dir("openai/whisper-large-v3-turbo"),
            torch_dtype=torch.float16,
            device=device,
        )
        # torch audio to numpy
        audio_numpy = audio_to_numpy(audio)
        # transformers.pipeline Dedicated Audio Format 1
        new_audio = {
            "sampling_rate": audio["sample_rate"], 
            "array": audio_numpy}
        prompt_text = pipe(
            new_audio,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
        return (audio,prompt_text,)


def temp_func(audio_path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_path = load_object_dir("openai/whisper-large-v3-turbo")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_path)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )
    dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    sample = dataset[0]["audio"]

    # result = pipe("audio.mp3")

    result = pipe(sample)
    print(result["text"])


NODE_CLASS_MAPPINGS = {
    "whisper_large_v3": whisper_large_v3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "whisper_large_v3": "Speech Recognition-whisper_large_v3",
}
