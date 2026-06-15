# syntax=docker/dockerfile:1
# ====================================================================================
# STAGE 1: The Builder  (Ubuntu 22.04 LTS)
# ====================================================================================
FROM ubuntu:22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# 1. Update, upgrade all system packages (patches vulnerabilities), then install build deps
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential autoconf automake libtool libtool-bin git wget unzip pkg-config \
    libjpeg-dev libsqlite3-dev libcurl4-openssl-dev libpcre3-dev \
    libspeex-dev libspeexdsp-dev libldns-dev libedit-dev libssl-dev \
    libopus-dev libsndfile1-dev python3-dev liblua5.2-dev \
    libtiff-dev yasm cmake libpq-dev uuid-dev zlib1g-dev ca-certificates \
    libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

# ------------------------------------------------------------------------------------
# 2. Build Libwebsockets v3.2 (required by mod_audio_fork)
# ------------------------------------------------------------------------------------
RUN git clone https://github.com/warmcat/libwebsockets.git -b v3.2-stable \
    && cd libwebsockets \
    && mkdir build && cd build \
    && cmake .. \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_BUILD_TYPE=Release \
        -DLWS_WITH_SSL=ON \
        -DLWS_WITHOUT_TESTAPPS=ON \
    && make -j$(nproc) \
    && make install \
    && ldconfig

# ------------------------------------------------------------------------------------
# 3. Build Telephony Dependencies
# ------------------------------------------------------------------------------------

# 3.1 LibKS
RUN git clone https://github.com/signalwire/libks.git libks \
    && cd libks \
    && cmake . -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) && make install && ldconfig

# 3.2 SignalWire-C
RUN git clone https://github.com/signalwire/signalwire-c.git signalwire-c \
    && cd signalwire-c \
    && cmake . -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release \
    && make -j$(nproc) && make install && ldconfig

# 3.3 Sofia-SIP
RUN git clone https://github.com/freeswitch/sofia-sip.git sofia-sip \
    && cd sofia-sip \
    && ./bootstrap.sh \
    && ./configure --prefix=/usr/local --with-pic --with-glib=no --without-doxygen --disable-stun \
    && make -j$(nproc) && make install && ldconfig

# 3.4 SpanDSP
RUN git clone https://github.com/freeswitch/spandsp.git spandsp \
    && cd spandsp \
    && git checkout 0d2e6ac \
    && ./bootstrap.sh \
    && ./configure --prefix=/usr/local --with-pic \
    && make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------------------------------
# 4. Build FreeSWITCH Core (mod_audio_fork is built separately in step 5)
# ------------------------------------------------------------------------------------
WORKDIR /usr/src
RUN git clone -b v1.10 https://github.com/signalwire/freeswitch.git freeswitch

WORKDIR /usr/src/freeswitch
RUN ./bootstrap.sh -j

# Enable mod_python3
RUN sed -i 's|#languages/mod_python3|languages/mod_python3|g' modules.conf

# Configure
RUN export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/pkgconfig \
    && export CFLAGS="-I/usr/local/include" \
    && export LDFLAGS="-L/usr/local/lib" \
    && ./configure \
        --prefix=/usr/local/freeswitch \
        --enable-core-pgsql-support \
        --with-python=$(which python3) \
        --disable-dependency-tracking

# Build and install
RUN make -j$(nproc) && make install

# Install sample configs, move to /conf for consistency
RUN make samples \
    && rm -rf /usr/local/freeswitch/conf \
    && mv /usr/local/freeswitch/etc/freeswitch /usr/local/freeswitch/conf

# Install sounds
RUN make cd-sounds-install cd-moh-install || true

# Write event_socket config (listen on all interfaces)
RUN echo '<configuration name="event_socket.conf" description="Socket Client">' > /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '  <settings>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="nat-map" value="false"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="listen-ip" value="0.0.0.0"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="listen-port" value="8021"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="password" value="ClueCon"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="apply-inbound-acl" value="loopback.auto"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '    <param name="stop-on-bind-error" value="true"/>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '  </settings>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml \
    && echo '</configuration>' >> /usr/local/freeswitch/conf/autoload_configs/event_socket.conf.xml

# Add mod_python3 to runtime modules
RUN sed -i 's|</modules>|  <load module="mod_python3"/>\n</modules>|' \
    /usr/local/freeswitch/conf/autoload_configs/modules.conf.xml

# ------------------------------------------------------------------------------------
# 5. Build mod_audio_fork separately
# ------------------------------------------------------------------------------------
WORKDIR /tmp/mod_audio_fork_build

RUN git clone --depth 1 https://github.com/connexcs/drachtio-freeswitch-modules-old.git

WORKDIR /tmp/mod_audio_fork_build/drachtio-freeswitch-modules-old/modules/mod_audio_fork

# Patch audio_pipe.cpp: remove lws_retry_bo_t struct (does not exist in libwebsockets v3.2)
RUN sed -i '/^static const lws_retry_bo_t retry = {/,/^};/d' audio_pipe.cpp \
    && sed -i '/retry_and_idle_policy/d' audio_pipe.cpp \
    && echo "patched audio_pipe.cpp"

# Patch parser.cpp: remove parse_ws_uri function (already defined in lws_glue.cpp)
RUN sed -i '/^switch_bool_t parse_ws_uri/,/^}/d' parser.cpp \
    && echo "patched parser.cpp"

# Compile all 4 source files into a shared library
RUN INC="-I/usr/src/freeswitch/src/include \
         -I/usr/src/freeswitch/libs/libteletone/src \
         -I/usr/local/include \
         -I." \
    && LIB="-L/usr/local/freeswitch/lib \
            -L/usr/local/lib \
            -lfreeswitch -lwebsockets -lssl -lcrypto -lpthread" \
    && RPH="-Wl,-rpath,/usr/local/lib,-rpath,/usr/local/freeswitch/lib" \
    && echo "Compiling mod_audio_fork.c..." \
    && gcc -fPIC -c mod_audio_fork.c $INC -o mod_audio_fork.o \
    && echo "Compiling audio_pipe.cpp..." \
    && g++ -fPIC -std=c++11 -c audio_pipe.cpp $INC -o audio_pipe.o \
    && echo "Compiling lws_glue.cpp..." \
    && g++ -fPIC -std=c++11 -c lws_glue.cpp $INC -o lws_glue.o \
    && echo "Compiling parser.cpp..." \
    && g++ -fPIC -std=c++11 -c parser.cpp $INC -o parser.o \
    && echo "Linking mod_audio_fork.so..." \
    && g++ -shared -o mod_audio_fork.so \
        mod_audio_fork.o audio_pipe.o lws_glue.o parser.o \
        $LIB $RPH \
    && echo "mod_audio_fork.so built successfully"

# Verify the module
RUN test $(stat -c%s mod_audio_fork.so) -gt 500000 || (echo "ERROR: module too small"; exit 1)
RUN nm mod_audio_fork.so | grep -c -i "AudioPipe" | xargs -I{} test {} -gt 10 || (echo "ERROR: missing AudioPipe symbols"; exit 1)

# Install into FreeSWITCH mod directory
RUN cp mod_audio_fork.so /usr/local/freeswitch/lib/freeswitch/mod/ \
    && chmod 755 /usr/local/freeswitch/lib/freeswitch/mod/mod_audio_fork.so

# Register in modules.conf.xml
RUN sed -i 's|</modules>|  <load module="mod_audio_fork"/>\n</modules>|' \
    /usr/local/freeswitch/conf/autoload_configs/modules.conf.xml

# ====================================================================================
# STAGE 2: Runtime Image  (Ubuntu 22.04 LTS — minimal)
# ====================================================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime libraries + upgrade all packages for security patches
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    libtiff5 libjpeg-turbo8 libsqlite3-0 libcurl4 libpcre3 \
    libspeex1 libspeexdsp1 libldns3 libedit2 libssl3 \
    libopus0 libsndfile1 liblua5.2-0 libpq5 uuid-runtime \
    ca-certificates libavformat58 libavcodec58 libswscale5 libavutil56 \
    libswresample3 netcat-openbsd procps \
    python3 libpython3.10 \
    && rm -rf /var/lib/apt/lists/*

# Copy FreeSWITCH and libraries from builder
COPY --from=builder /usr/local/freeswitch /usr/local/freeswitch
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include

RUN ldconfig

# Create freeswitch user
RUN groupadd -r freeswitch \
    && useradd -r -g freeswitch -d /usr/local/freeswitch -s /bin/bash freeswitch \
    && chown -R freeswitch:freeswitch /usr/local/freeswitch

ENV PATH="/usr/local/freeswitch/bin:${PATH}"
WORKDIR /usr/local/freeswitch

EXPOSE 5060/udp 5060/tcp 5080/udp 5080/tcp 8021/tcp 5061/udp 5061/tcp
EXPOSE 16384-16484/udp

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD test -f /usr/local/freeswitch/run/freeswitch.pid || exit 1

CMD ["freeswitch", "-nonat", "-nf", "-conf", "/usr/local/freeswitch/conf", "-log", "/usr/local/freeswitch/log", "-db", "/usr/local/freeswitch/db", "-run", "/usr/local/freeswitch/run", "-c"]