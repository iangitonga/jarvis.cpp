import argparse
import os
import urllib

import torch
from safetensors import safe_open


MODELS_INFO = {
    "smollm2-sm": {
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/model.safetensors",
        "num_hidden_layers": 30
    },
    "smollm2-md": {
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/model.safetensors",
        "num_hidden_layers": 32
    },
    "smollm2-lg": {
        "url": "https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct/resolve/main/model.safetensors",
        "num_hidden_layers": 24
    }
}



def _download_model(url, model_path):
    print(f"Downloading {model_path}")
    with urllib.request.urlopen(url) as source, open(model_path, "wb") as output:
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
            # trick to make it work with jupyter.
            print(f"\rDownload Progress [{downloaded_mb}/{download_size_mb}MB]: {progress_perc}%", end="")
    print("\n\n")



def write_tensor(name, checkpoint, fout):
    tensor = checkpoint.get_tensor(name)
    tensor = tensor.to(torch.float16)
    tensor_bytes = tensor.numpy().tobytes()
    fout.write(tensor_bytes)


def convert_smollm2(model_name, in_model_path):
    print(f"Converting {model_name} to dtype: Float16")

    if not in_model_path or not os.path.isfile(in_model_path):
        in_model_path = f"{model_name}.safetensors"
        _download_model(MODELS_INFO[model_name]["url"], in_model_path)

    out_model_path = f"{model_name}.bin"
    with safe_open(in_model_path, framework="pt", device="cpu") as ckpt, open(out_model_path, "wb") as fout:
        print("Converting ...")

        magic_no = b"smollm2f"
        fout.write(magic_no)

        write_tensor("model.embed_tokens.weight", ckpt, fout)
        
        num_layers = MODELS_INFO[model_name]["num_hidden_layers"]
        for i in range(num_layers):
            write_tensor(f"model.layers.{i}.input_layernorm.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.self_attn.q_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.self_attn.k_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.self_attn.v_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.self_attn.o_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.post_attention_layernorm.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.mlp.gate_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.mlp.up_proj.weight", ckpt, fout)
            write_tensor(f"model.layers.{i}.mlp.down_proj.weight", ckpt, fout)

        write_tensor("model.norm.weight", ckpt, fout)
    print(f"\nConversion complete: {out_model_path}")



parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="Model size.", choices=MODELS_INFO.keys())
parser.add_argument("--model_path", help="Path of the model if cached.")

args = parser.parse_args()
convert_smollm2(args.model_name, args.model_path)
