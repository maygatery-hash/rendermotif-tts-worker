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

COPY handler.py /content/handler.py

CMD ["python", "-u", "/content/handler.py"]