import base64
import gc
import os
import subprocess
import numpy as np
import pyloudnorm as pyln
import scipy.signal as signal
import soundfile as sf
import torch

import runpod
from qwen_tts import Qwen3TTSModel

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("🚀 Pre-loading Qwen3-TTS Base Model into VRAM...")
model_base = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map=device, dtype=dtype
)


def apply_production_mastering(audio_data, sample_rate, target_lufs=-16.0):
  audio = audio_data.astype(np.float32)
  b, a = signal.butter(4, 80.0 / (sample_rate / 2.0), btype="high")
  audio_filtered = signal.filtfilt(b, a, audio)

  peak = np.max(np.abs(audio_filtered))
  if peak > 0.85:
    audio_filtered = audio_filtered * (0.85 / peak)

  meter = pyln.Meter(sample_rate)
  current_lufs = meter.integrated_loudness(audio_filtered)

  if current_lufs > -70.0:
    normalized = pyln.normalize.loudness(
        audio_filtered, current_lufs, target_lufs
    )
  else:
    normalized = audio_filtered

  max_peak = np.max(np.abs(normalized))
  if max_peak > 0.94:
    normalized = normalized * (0.94 / max_peak)

  return normalized, current_lufs


def handler(job):
  job_input = job["input"]

  script_json = job_input.get("script_json", [])
  anchor_text = job_input.get("anchor_text", "")
  anchor_wav_base64 = job_input.get("anchor_wav_base64", None)
  batch_size = job_input.get("batch_size", 8)

  if not script_json or not anchor_text or not anchor_wav_base64:
    return {
        "error": (
            "Missing required parameters: script_json, anchor_text, or"
            " anchor_wav_base64"
        )
    }

  anchor_path = "/tmp/anchor.wav"

  with open(anchor_path, "wb") as f:
    f.write(base64.b64decode(anchor_wav_base64))

  prompt_items = model_base.create_voice_clone_prompt(
      ref_audio=anchor_path,
      ref_text=anchor_text,
      x_vector_only_mode=False,
  )

  total_acts = len(script_json)
  audio_segments = [None] * total_acts

  with torch.inference_mode():
    for batch_start in range(0, total_acts, batch_size):
      batch_end = min(batch_start + batch_size, total_acts)
      batch_items = script_json[batch_start:batch_end]

      texts = [item["processed_text"] for item in batch_items]
      temp = batch_items[0].get("recommended_temperature", 0.75)

      torch.manual_seed(42 + batch_start)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42 + batch_start)

      wavs, sr = model_base.generate_voice_clone(
          text=texts,
          language="English",
          voice_clone_prompt=prompt_items,
          temperature=temp,
      )

      for i, wav in enumerate(wavs):
        act_index = batch_start + i
        audio_segments[act_index] = wav

      gc.collect()
      if torch.cuda.is_available():
        torch.cuda.empty_cache()

  stitched_chunks = []
  for wav in audio_segments:
    stitched_chunks.append(wav)
    silence = np.zeros(int(sr * 0.50), dtype=wav.dtype)
    stitched_chunks.append(silence)

  raw_master = np.concatenate(stitched_chunks)
  mastered_audio, _ = apply_production_mastering(
      raw_master, sr, target_lufs=-16.0
  )

  master_wav_path = "/tmp/output_master.wav"
  master_mp3_path = "/tmp/output_master.mp3"

  sf.write(master_wav_path, mastered_audio, sr)

  subprocess.run(
      [
          "ffmpeg",
          "-y",
          "-i",
          master_wav_path,
          "-codec:a",
          "libmp3lame",
          "-qscale:a",
          "2",
          master_mp3_path,
      ],
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL,
  )

  with open(master_mp3_path, "rb") as f:
    mp3_base64 = base64.b64encode(f.read()).decode("utf-8")

  for p in [anchor_path, master_wav_path, master_mp3_path]:
    if os.path.exists(p):
      os.remove(p)

  return {"status": "success", "audio_mp3_base64": mp3_base64}


runpod.serverless.start({"handler": handler})