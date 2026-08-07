FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /content

RUN apt-get update && apt-get install -y \
    sox \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    qwen-tts \
    pyloudnorm \
    soundfile \
    scipy \
    torchaudio \
    hf_transfer

# Pre-compiled wheel installs in 5 seconds with zero C++ compile overhead
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

COPY handler.py /content/handler.py

CMD ["python", "-u", "/content/handler.py"]