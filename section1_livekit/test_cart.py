"""Quick standalone Cartesia check — run: python test_cartesia.py
Writes out.wav if synthesis works; prints the raw API error if not."""
import os

import httpx

from env_loader import load_env

load_env()

API_KEY = os.getenv("CARTESIA_API_KEY", "")
MODEL = os.getenv("CARTESIA_MODEL", "sonic-2")
# Cartesia's public default voice (Barbershop Man). Any valid voice works.
VOICE_ID = os.getenv("CARTESIA_VOICE_ID") or "a0e99841-438c-4a64-b679-ae501e7d6091"

print(f"key: {API_KEY[:10]}...  model: {MODEL}  voice: {VOICE_ID}")

resp = httpx.post(
    "https://api.cartesia.ai/tts/bytes",
    headers={
        "X-API-Key": API_KEY,
        "Cartesia-Version": "2024-06-10",
        "Content-Type": "application/json",
    },
    json={
        "model_id": MODEL,
        "transcript": "Hello, your order is out for delivery.",
        "voice": {"mode": "id", "id": VOICE_ID},
        "output_format": {
            "container": "wav",
            "encoding": "pcm_s16le",
            "sample_rate": 24000,
        },
    },
    timeout=30,
)

print("status:", resp.status_code)
if resp.status_code == 200:
    with open("out.wav", "wb") as f:
        f.write(resp.content)
    print(f"OK — wrote out.wav ({len(resp.content)} bytes). Play it to confirm.")
else:
    print("ERROR BODY:", resp.text)