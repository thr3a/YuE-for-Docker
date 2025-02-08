FROM --platform=linux/x86_64 nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ARG PACKAGES="git curl ca-certificates vim wget unzip build-essential cmake jq"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=on
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_PROJECT_ENVIRONMENT="/usr/"
ENV UV_LINK_MODE=copy

RUN apt-key adv --keyserver keyserver.ubuntu.com --recv f23c5a6cf475977595c89f51ba6932366a755776 \
 && echo "deb http://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/python.list \
 && echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" >> /etc/apt/sources.list.d/python.list

RUN apt-get update \
 && apt-get install -y --no-install-recommends ${PACKAGES} python${PYTHON_VERSION} \
 && ln -nfs /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
 && ln -nfs /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 \
 && rm -rf /var/lib/apt/lists/* \
 && curl -sSL https://bootstrap.pypa.io/get-pip.py | python - \
 && pip install uv

WORKDIR /app

COPY ./pyproject.toml ./
# COPY ./requirements-uv.txt ./
COPY ./uv.lock ./

# RUN uv pip sync requirements-uv.txt --index-strategy unsafe-best-match --system --index https://download.pytorch.org/whl/cu124
RUN uv sync --frozen --no-cache
RUN uv run huggingface-cli download m-a-p/xcodec_mini_infer --local-dir inference/xcodec_mini_infer

COPY . ./
