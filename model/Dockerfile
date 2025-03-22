FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /home/app

COPY . .

RUN apt-get update && apt-get upgrade && \
apt-get install ffmpeg libsm6 libxext6 -y && \
uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 && \
uv pip install --system kagglehub notebook python-dotenv ultralytics

ENTRYPOINT [ "uv", "run", "./pre-alpha_model.py" ]

VOLUME /data