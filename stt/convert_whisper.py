import argparse
import hashlib
import urllib
import warnings
import os

import torch
from tqdm import tqdm



MAGIC_NUMBER = b"whisperf"
MODELS_URLS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
}


def write_tensor(f, tensor):
    tensor = tensor.to(torch.float16)
    tensor_bytes = tensor.numpy().tobytes()
    f.write(tensor_bytes)

    
def write_encoder(f, ckpt):
    c = ckpt["model_state_dict"]
    d = ckpt["dims"]

    write_tensor(f, c["encoder.conv1.weight"].transpose(1, 2))
    write_tensor(f, c["encoder.conv1.bias"])
    write_tensor(f, c["encoder.conv2.weight"].transpose(1, 2))
    write_tensor(f, c["encoder.conv2.bias"])
    write_tensor(f, c["encoder.positional_embedding"])
    
    for i in range(d["n_audio_layer"]):
        write_tensor(f, c[f"encoder.blocks.{i}.attn_ln.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn_ln.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.query.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.query.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.key.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.value.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.value.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.out.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.attn.out.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp_ln.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp_ln.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp.0.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp.0.bias"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp.2.weight"])
        write_tensor(f, c[f"encoder.blocks.{i}.mlp.2.bias"])
    
    write_tensor(f, c["encoder.ln_post.weight"])
    write_tensor(f, c["encoder.ln_post.bias"])
    

def write_decoder(f, ckpt):
    c = ckpt["model_state_dict"]
    d = ckpt["dims"]

    write_tensor(f, c[f"decoder.token_embedding.weight"])
    write_tensor(f, c[f"decoder.positional_embedding"])
    
    for i in range(d["n_text_layer"]):
        write_tensor(f, c[f"decoder.blocks.{i}.attn_ln.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn_ln.bias"])

        write_tensor(f, c[f"decoder.blocks.{i}.attn.query.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.query.bias"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.key.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.value.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.value.bias"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.out.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.attn.out.bias"])

        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn_ln.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn_ln.bias"])

        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.query.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.query.bias"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.key.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.value.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.value.bias"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.out.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.cross_attn.out.bias"])

        write_tensor(f, c[f"decoder.blocks.{i}.mlp_ln.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.mlp_ln.bias"])

        write_tensor(f, c[f"decoder.blocks.{i}.mlp.0.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.mlp.0.bias"])
        write_tensor(f, c[f"decoder.blocks.{i}.mlp.2.weight"])
        write_tensor(f, c[f"decoder.blocks.{i}.mlp.2.bias"])
    
    write_tensor(f, c["decoder.ln.weight"])
    write_tensor(f, c["decoder.ln.bias"])


def download_model(url, root):
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")

    return download_target



def convert_model(model_name, model_path):
    if not model_path:
        model_path = download_model(MODELS_URLS[model_name], ".")
    print(f"Converting: {model_path}")

    with open(model_path, "rb") as fp:
        ckpt = torch.load(fp, weights_only=True, map_location="cpu")

    with open(f"whisper-{model_name}.bin", "wb") as f:
        f.write(MAGIC_NUMBER)
        print("Converting encoder...")
        write_encoder(f, ckpt)
        print("Converting decoder...")
        write_decoder(f, ckpt)
    print("Conversion complete.")

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODELS_URLS.keys())
parser.add_argument("--path", default="", help="Optional path to source model if you have it locally.")

args = parser.parse_args()

convert_model(args.model, args.path)
