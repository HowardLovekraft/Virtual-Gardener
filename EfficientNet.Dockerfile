FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /app

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./pyproject.toml ./

RUN uv venv && \
	uv pip install -r pyproject.toml

COPY ./papka ./papka
COPY ./model/ ./

ENTRYPOINT [ "uv", "run", "./train_eff_net.py" ]

VOLUME /data
