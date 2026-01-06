ARG BUILD_FROM
FROM ${BUILD_FROM}

EXPOSE 10300
VOLUME /data
ENV MODEL=rnnt

# Install gigaAM
WORKDIR /usr/src
ARG WYOMING_GIGAAM_VERSION

RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        netcat-traditional \
        python3 \
        python3-dev \
        python3-pip \
        git \
    \
    && pip3 install --no-cache-dir --break-system-packages -U \
        setuptools \
        wheel \
    && pip3 install --no-cache-dir --break-system-packages \
        "wyoming-gigaAM @ https://github.com/yusinv/wyoming-GigaAM/archive/refs/tags/v${WYOMING_GIGAAM_VERSION}.tar.gz" \
    \
    && apt-get purge -y --auto-remove \
        build-essential \
        python3-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

ENTRYPOINT exec wyoming-gigaAM --model ${MODEL} --uri tcp://0.0.0.0:10300 --data-dir /data

HEALTHCHECK --start-period=10m \
    CMD echo '{ "type": "describe" }' \
        | nc -w 1 localhost 10300 \
        | grep -q "GigaAM" \
        || exit 1