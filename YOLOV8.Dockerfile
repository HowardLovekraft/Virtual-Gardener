FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt-get update && \
    apt-get install ffmpeg libsm6 libxext6 curl unzip -y && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ./model/ ./pyproject.toml ./

RUN --mount=type=secret,id=ROBO_LINK,env=ROBO_LINK \
	curl -L "https://app.roboflow.com/ds/$ROBO_LINK" > roboflow.zip && \
    mkdir papka && \
    unzip roboflow.zip -d ./papka/ && \
	mv ./papka/valid ./papka/val && \
    rm roboflow.zip && \
	uv venv && \
	uv pip install -r pyproject.toml

ENTRYPOINT [ "uv", "run", "./train_yolo08.py" ]

VOLUME /data