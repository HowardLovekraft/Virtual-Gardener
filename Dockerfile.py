FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/* # Очистка кэша APT для уменьшения размера образа

COPY py_server/requirements.txt ./py_server/
RUN UV_HTTP_TIMEOUT=300 uv pip install --system -r py_server/requirements.txt

COPY . .

ENTRYPOINT [ "uvicorn", "py_server.server:app", "--host", "0.0.0.0", "--port", "8000" ]

VOLUME /data