FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /home/app

COPY ./model/ .

RUN apt-get update && apt-get upgrade && \
apt-get install ffmpeg libsm6 libxext6 -y && \
uv pip install --system -r requirements.txt

ENTRYPOINT [ "uv", "run", "./train_model.py" ]

VOLUME /data