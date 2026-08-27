# ── Stage 1: build SWI-Prolog + janus_swi from source ──────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git \
    libssl-dev libgmp-dev libarchive-dev \
    libpcre2-dev libedit-dev libossp-uuid-dev \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

ARG SWIPL_VERSION=10.0.2-0-jammyppa2
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:swi-prolog/stable && \
    apt-get update && apt-get install -y swi-prolog=${SWIPL_VERSION} && \
    rm -rf /var/lib/apt/lists/*

ENV SWIPY_COMMIT=9cdad0186073f01afb2c011908f849bc0f52029f
RUN git clone https://github.com/SWI-Prolog/packages-swipy /tmp/packages-swipy && \
    cd /tmp/packages-swipy && \
    git checkout ${SWIPY_COMMIT} && \
    pip3 install .

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# PeTTa is held at e149089. Commits after it break PeTTaChainer's backward
# chainer: query() returns [] while forward chaining still succeeds, failing
# 5 of 8 upstream chainer tests. Verified by pairwise bisect — this pin with
# chainer 02a85c6 passes 8/8. Do not bump without re-running that suite.
ENV PETTA_COMMIT=e1490899cefc67c128d5311ff4861f9997674957
ENV PETTACHAINER_COMMIT=02a85c63be6a735f50f50e1084a717b3256b8406

ARG SWIPL_VERSION=10.0.2-0-jammyppa2
RUN apt-get update && apt-get install -y \
    software-properties-common \
    python3 python3-pip \
    libgmp10 libarchive13 libpcre2-8-0 libedit2 \
    git \
    && add-apt-repository ppa:swi-prolog/stable \
    && apt-get update && apt-get install -y swi-prolog=${SWIPL_VERSION} \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Clone PeTTa and PeTTaChainer
WORKDIR /deps
# PETTA_COMMIT is not reachable from any PeTTa branch head, so a plain clone
# cannot check it out ("reference is not a tree"). Fetch it by SHA explicitly.
RUN git clone https://github.com/trueagi-io/PeTTa.git && \
    git -C /deps/PeTTa fetch origin ${PETTA_COMMIT} && \
    git -C /deps/PeTTa checkout ${PETTA_COMMIT} && \
    git clone https://github.com/rTreutlein/PeTTaChainer.git && \
    git -C /deps/PeTTaChainer checkout ${PETTACHAINER_COMMIT}

# Install PeTTa — strip the janus-swi pip dep, then point PYTHONPATH
# at the actual source dir so `import petta` resolves correctly
RUN cd /deps/PeTTa && \
    sed -i "/'janus-swi'/d" setup.py && \
    pip3 install .

ENV PYTHONPATH=/deps/PeTTa/python:$PYTHONPATH

# Install PeTTaChainer
RUN cd /deps/PeTTaChainer && \
    pip3 install -e .

# Install pln-rag app dependencies
WORKDIR /app
COPY requirements.txt .
RUN rm -rf /usr/lib/python3/dist-packages/blinker*
RUN pip3 install --default-timeout=1000 -r requirements.txt

COPY . .

VOLUME ["/app/data"]

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
