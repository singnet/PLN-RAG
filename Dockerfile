# ── Stage 1: build SWI-Prolog + janus_swi from source ──────────────────────
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build git \
    libssl-dev libgmp-dev libarchive-dev \
    libpcre2-dev libedit-dev libossp-uuid-dev \
    python3 python3-pip python3-dev ca-certificates curl gnupg \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x4AB3A5F60EA9AEB3" \
        -o /tmp/swi-prolog-ppa.asc && \
    test "$(gpg --show-keys --with-colons /tmp/swi-prolog-ppa.asc | awk -F: '$1 == "fpr" {print $10; exit}')" \
        = "E8B739E3753FF4A12360BA6A4AB3A5F60EA9AEB3" && \
    gpg --batch --dearmor --output /usr/share/keyrings/swi-prolog-ppa.gpg \
        /tmp/swi-prolog-ppa.asc && \
    echo "deb [signed-by=/usr/share/keyrings/swi-prolog-ppa.gpg] https://ppa.launchpadcontent.net/swi-prolog/stable/ubuntu jammy main" \
        > /etc/apt/sources.list.d/swi-prolog-ppa.list && \
    rm /tmp/swi-prolog-ppa.asc && \
    apt-get update && apt-get install -y swi-prolog && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/SWI-Prolog/packages-swipy /tmp/packages-swipy && \
    cd /tmp/packages-swipy && \
    pip3 install .

# ── Stage 2: runtime image ──────────────────────────────────────────────────
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PETTA_COMMIT=e1490899cefc67c128d5311ff4861f9997674957
ENV PETTACHAINER_COMMIT=02a85c63be6a735f50f50e1084a717b3256b8406

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libgmp10 libarchive13 libpcre2-8-0 libedit2 \
    git ca-certificates curl gnupg \
    && curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x4AB3A5F60EA9AEB3" \
        -o /tmp/swi-prolog-ppa.asc \
    && test "$(gpg --show-keys --with-colons /tmp/swi-prolog-ppa.asc | awk -F: '$1 == "fpr" {print $10; exit}')" \
        = "E8B739E3753FF4A12360BA6A4AB3A5F60EA9AEB3" \
    && gpg --batch --dearmor --output /usr/share/keyrings/swi-prolog-ppa.gpg \
        /tmp/swi-prolog-ppa.asc \
    && echo "deb [signed-by=/usr/share/keyrings/swi-prolog-ppa.gpg] https://ppa.launchpadcontent.net/swi-prolog/stable/ubuntu jammy main" \
        > /etc/apt/sources.list.d/swi-prolog-ppa.list \
    && rm /tmp/swi-prolog-ppa.asc \
    && apt-get update && apt-get install -y swi-prolog \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages

# Clone PeTTa and PeTTaChainer
WORKDIR /deps
RUN git clone https://github.com/trueagi-io/PeTTa.git && \
    (git -C /deps/PeTTa checkout ${PETTA_COMMIT} || \
     (git -C /deps/PeTTa fetch origin ${PETTA_COMMIT} && \
      git -C /deps/PeTTa checkout FETCH_HEAD)) && \
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
