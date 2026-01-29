import io
import os
import glob
import time
import sys
import torch
import numpy as np
import soundfile as sf
import jieba
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# --- FIX 1: Force UTF-8 encoding ---
sys.stdout.reconfigure(encoding='utf-8')

# 1. Setup local paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")

# --- FIX 2: 重定向 jieba 缓存到 python3 目录下，保持根目录整洁 ---
# 如果 python3 目录不存在，它会回退到 BASE_DIR，你也可以改为 os.path.join(BASE_DIR, "cache")
CACHE_DIR = os.path.join(BASE_DIR, "python3")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)
jieba.dt.tmp_dir = CACHE_DIR
jieba.dt.cache_file = os.path.join(CACHE_DIR, "jieba.cache")

# --- FIX 3: Misaki API Compatibility Patch ---
try:
    import misaki.espeak
    import espeakng_loader
    def set_data_path_patch(path):
        misaki.espeak.EspeakWrapper.data_path = path
    # Attach the patch for newer misaki versions
    misaki.espeak.EspeakWrapper.set_data_path = staticmethod(set_data_path_patch)
    misaki.espeak.EspeakWrapper.set_data_path(espeakng_loader.get_data_path())
except Exception:
    pass

from kokoro import KPipeline

# --- Startup Timing Begins ---
boot_start = time.perf_counter()
print(">>>[1/3] INITIALIZING GCETTS SERVER (GPU MODE)...", flush=True)

def log_step(message, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"    Done: {message} ({elapsed:.2f}s)", flush=True)

# --- Hardware Acceleration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    print(f">>> GPU ACTIVE: {torch.cuda.get_device_name(0)}", flush=True)

app = FastAPI(title="Kokoro Local API")

# 2. Pre-load pipelines
print(f">>> [2/3] LOADING MODELS TO {device.upper()}...", flush=True)
pipeline_start = time.perf_counter()
try:
    pipelines = {
        'a': KPipeline(lang_code='a', device=device, repo_id='hexgrad/Kokoro-82M'),
        'z': KPipeline(lang_code='z', device=device, repo_id='hexgrad/Kokoro-82M')
    }
    log_step(f"AI Pipelines Ready ({device.upper()})", pipeline_start)
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to load pipelines: {e}", flush=True)
    raise

print(f">>> [3/3] BOOT COMPLETE. Total Time: {time.perf_counter() - boot_start:.2f}s", flush=True)
print(">>> LISTENING ON: http://0.0.0.0:8880", flush=True)
print("-" * 50, flush=True)

class TTSRequest(BaseModel):
    input: str
    voice: str = "af_heart"
    speed: float = 1.0

@app.get("/v1/audio/voices")
async def get_voices():
    if not os.path.exists(VOICES_DIR):
        return {"voices": []}
    voice_files = glob.glob(os.path.join(VOICES_DIR, "*.pt"))
    voices = [os.path.splitext(os.path.basename(f))[0] for f in voice_files]
    return {"voices": sorted(voices)}

@app.post("/v1/audio/speech")
async def speech(request: TTSRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail="Input is empty")
    
    request_start = time.perf_counter()
    try:
        # Determine language (z = Chinese)
        lang_code = 'z' if request.voice.lower().startswith('z') else 'a'
        pipeline = pipelines.get(lang_code, pipelines['a'])
        
        # Use local voice if available
        voice_path = os.path.join(VOICES_DIR, f"{request.voice}.pt")
        voice_to_use = voice_path if os.path.exists(voice_path) else request.voice

        print(f"REQ: [{request.voice}] {request.input[:30]}...", flush=True)

        # AI Generation
        gen_start = time.perf_counter()
        generator = pipeline(request.input, voice=voice_to_use, speed=request.speed)
        audio_list = [audio for _, _, audio in generator]
        
        if not audio_list: raise Exception("No audio generated")
        final_audio = np.concatenate(audio_list)
        gen_duration = time.perf_counter() - gen_start

        # Audio encoding
        enc_start = time.perf_counter()
        buffer = io.BytesIO()
        sf.write(buffer, final_audio, 24000, format='WAV', subtype='PCM_16')
        buffer.seek(0)
        enc_duration = time.perf_counter() - enc_start

        print(f"RES: Gen {gen_duration:.2f}s | Enc {enc_duration:.2f}s | Total {time.perf_counter()-request_start:.2f}s", flush=True)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        print(f"ERROR: {str(e)}", flush=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880, log_level="error")