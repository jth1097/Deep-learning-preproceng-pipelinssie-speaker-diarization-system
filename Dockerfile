# syntax=docker/dockerfile:1.6
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libboost-all-dev \
        libbz2-dev \
        liblzma-dev \
        zlib1g-dev \
        ffmpeg \
        sox \
        libsox-dev \
        libsndfile1 \
        libsndfile1-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt ./

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/workspace

RUN mkdir -p logs diarization_output vad_output_frames whisper_output_frames

ENTRYPOINT ["python", "scripts/run_diar_experiment.py"]
CMD ["--help"]
