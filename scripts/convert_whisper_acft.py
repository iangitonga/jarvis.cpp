"""
Converts whisper models finetuned by futo-org to allow decoding input audio with dynamic length (audio_ctx).

More info at: https://github.com/futo-org/whisper-acft/tree/main
"""


import argparse
import urllib
import os

import torch
from safetensors import safe_open
from tqdm import tqdm



MAGIC_NUMBER = b"whisperf"
MODELS_URLS = {
    "tiny.en": "https://huggingface.co/futo-org/acft-whisper-tiny.en/resolve/main/model.safetensors",
    "base.en": "https://huggingface.co/futo-org/acft-whisper-base.en/resolve/main/model.safetensors",
    "small.en": "https://huggingface.co/futo-org/acft-whisper-small.en/resolve/main/model.safetensors",
}


def write_tensor(f, tensor):
    tensor = tensor.to(torch.float16)
    tensor_bytes = tensor.numpy().tobytes()
    f.write(tensor_bytes)

    
def write_encoder(f, ckpt, n_layers):
    write_tensor(f, ckpt["model.encoder.conv1.weight"].transpose(1, 2))
    write_tensor(f, ckpt["model.encoder.conv1.bias"])
    write_tensor(f, ckpt["model.encoder.conv2.weight"].transpose(1, 2))
    write_tensor(f, ckpt["model.encoder.conv2.bias"])
    write_tensor(f, ckpt["model.encoder.embed_positions.weight"])
    
    for i in range(n_layers):
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn_layer_norm.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn_layer_norm.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.q_proj.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.q_proj.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.k_proj.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.v_proj.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.v_proj.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.out_proj.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.self_attn.out_proj.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.final_layer_norm.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.final_layer_norm.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.fc1.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.fc1.bias"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.fc2.weight"])
        write_tensor(f, ckpt[f"model.encoder.layers.{i}.fc2.bias"])
    
    write_tensor(f, ckpt["model.encoder.layer_norm.weight"])
    write_tensor(f, ckpt["model.encoder.layer_norm.bias"])


def write_decoder(f, ckpt, n_layers):
    write_tensor(f, ckpt[f"model.decoder.embed_tokens.weight"])
    write_tensor(f, ckpt[f"model.decoder.embed_positions.weight"])
    
    for i in range(n_layers):
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn_layer_norm.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn_layer_norm.bias"])

        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.q_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.q_proj.bias"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.k_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.v_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.v_proj.bias"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.out_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.self_attn.out_proj.bias"])

        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias"])

        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.q_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.q_proj.bias"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.k_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.v_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.v_proj.bias"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.out_proj.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.encoder_attn.out_proj.bias"])

        write_tensor(f, ckpt[f"model.decoder.layers.{i}.final_layer_norm.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.final_layer_norm.bias"])

        write_tensor(f, ckpt[f"model.decoder.layers.{i}.fc1.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.fc1.bias"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.fc2.weight"])
        write_tensor(f, ckpt[f"model.decoder.layers.{i}.fc2.bias"])
    
    write_tensor(f, ckpt["model.decoder.layer_norm.weight"])
    write_tensor(f, ckpt["model.decoder.layer_norm.bias"])


def download_model(url, root):
    os.makedirs(root, exist_ok=True)

    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return download_target



def convert_model(model_name, model_path):
    if not model_path:
        model_path = download_model(MODELS_URLS[model_name], ".")
    print(f"Converting: {model_path}")

    ckpt = {}
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            ckpt[k] = f.get_tensor(k)
    
    n_embd = ckpt["model.encoder.layer_norm.weight"].shape[0]
    if (n_embd == 384): n_layers = 4
    elif (n_embd == 512): n_layers = 6
    elif (n_embd == 768): n_layers = 12
    else: raise Exception

    with open(f"acft-whisper-{model_name}.bin", "wb") as f:
        f.write(MAGIC_NUMBER)
        print("Converting encoder...")
        write_encoder(f, ckpt, n_layers)
        print("Converting decoder...")
        write_decoder(f, ckpt, n_layers)
    print("Conversion complete.")

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODELS_URLS.keys())
parser.add_argument("--path", default="", help="Optional path to source model if you have it locally.")

args = parser.parse_args()

convert_model(args.model, args.path)

