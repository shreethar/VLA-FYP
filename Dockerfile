FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    git build-essential curl wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv

RUN pip install --upgrade pip setuptools wheel

COPY requirements_.txt /tmp/requirements_.txt
RUN pip install -r /tmp/requirements_.txt

RUN pip install jupyterlab ipykernel
