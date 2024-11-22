#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "stt": {
        "whisper-tiny.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-tiny.en.bin",
        "whisper-base.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-base.en.bin",
        "whisper-small.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-small.en.bin",
        "whisper-medium.en.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/stt/whisper-medium.en.bin"
    },
    "llm": {
        "smollm2-sm.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-sm.bin",
        "smollm2-md.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-md.bin",
        "smollm2-lg.bin": "https://huggingface.co/iangitonga/jarvis.cpp/resolve/main/llm/smollm2-lg.bin"
    }
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


def download_model(stt_model_path, llm_model_path):
    os.makedirs("models", exist_ok=True)
    if not os.path.exists(stt_model_path):
        stt_model_name = stt_model_path.split("/")[1]
        stt_model_url = MODELS_URLS["stt"][stt_model_name]
        print("Downloading stt model:")
        try:
            _download_model(stt_model_url, stt_model_path)
        except: # Delete if file uncomplete.
            if os.path.isfile(stt_model_path):
                os.remove()
            raise
    if not os.path.exists(llm_model_path):
        llm_model_name = llm_model_path.split("/")[1]
        llm_model_url = MODELS_URLS["llm"][llm_model_name]
        print("Downloading llm model:")
        try:
            _download_model(llm_model_url, llm_model_path)
        except: # Delete if file uncomplete.
            if os.path.isfile(llm_model_path):
                os.remove()
            raise
    print()


if len(sys.argv) != 3:
    print(f"Args provided: {sys.argv}")
    exit(-1)

try:
    download_model(sys.argv[1], sys.argv[2])
except Exception as e:
    # print(e.with_traceback())
    exit(-2)
