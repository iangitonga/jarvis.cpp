#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "whisper-tiny.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-tiny.en.bin",
    "whisper-base.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-base.en.bin",
    "whisper-small.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-small.en.bin",
    "whisper-medium.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-medium.en.bin",

    "acft-whisper-tiny.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/acft-whisper-tiny.en.bin",
    "acft-whisper-base.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/acft-whisper-base.en.bin",
    "acft-whisper-small.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/acft-whisper-small.en.bin",

    "smollm2-sm.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-sm.bin",
    "smollm2-md.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-md.bin",
    "smollm2-lg.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-lg.bin"
}


def _download_model(url: str, model_path: str):
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        download_size_mb = int(download_size / 1000_000)
        downloaded = 0
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            downloaded += len(buffer)
            progress_perc = int(downloaded / download_size * 100.0)
            downloaded_mb = int(downloaded / 1000_000)
            print(f"\rDownload Progress [{downloaded_mb}/{download_size_mb}MB]: {progress_perc}%", end="")
    print()


def download_model(model_names):
    os.makedirs("models", exist_ok=True)

    for model_name in model_names:
        model_path = os.path.join("models", model_name)
        # TODO: Check if the model is complete i.e in case download failed halfway.
        if not os.path.exists(model_path):
            model_name = model_path.split("/")[1]
            model_url = MODELS_URLS[model_name]
            print(f"Downloading model: {model_name}")
            _download_model(model_url, model_path)
            print()

if len(sys.argv) < 2:
    print(f"Args provided: {sys.argv}")
    exit(-1)

try:
    download_model(sys.argv[1:])
except Exception as e:
    # print(e.with_traceback())
    exit(-2)
