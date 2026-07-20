# Slim CPU image. Runs the mock backend out of the box (docker run works with
# zero model download). For a real model, set MODEL_BACKEND=hf and MODEL_ID, and
# base off an image with the model deps / GPU runtime (e.g. nvidia/cuda + torch).
FROM python:3.11-slim

WORKDIR /app

# Only the serving deps are needed for the mock backend; the heavy ML deps are
# commented so the default image stays small and builds fast.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV MODEL_BACKEND=mock \
    MAX_NEW_TOKENS=128

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
