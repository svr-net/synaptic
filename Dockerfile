# syntax=docker/dockerfile:1
#
# Multi-stage build of the learning-theory library, its WebAssembly module and the web front end.
#
#   docker build -t synaptic .                    # standalone web front end on nginx, default target
#   docker run --rm -p 8080:80 synaptic           # -> http://localhost:8080
#
#   docker build --target native -t synaptic:native .   # C++ build + unit tests
#   docker run --rm synaptic:native                     # runs the end-to-end demo
#
#   docker build --target e2e .                     # browser tests of every page (Playwright,
#                                                   # WebGPU on SwiftShader);
#                                                   # behind a TLS-intercepting proxy add
#                                                   # --secret id=ca,src=<proxy CA bundle>
#   docker build --target wasm-artifacts --output web/wasm .   # export syn.js/syn.wasm to the host
#   docker build --target standalone-artifacts --output dist .  # export the standalone site + zip
#
# Every build stage runs its tests, so a successful build means they passed.

ARG EMSDK_VERSION=4.0.10
ARG PLAYWRIGHT_VERSION=1.56.1
ARG NODE_VERSION=22

# Layers are ordered so that each depends only on its inputs: toolchains and npm packages first,
# then the C++ sources for the compiles, and the web pages only where they are used (an edit to a
# page does not recompile the library). The compiles also keep ccache in BuildKit cache mounts,
# so a change to one source file recompiles that file only.

# ---------------------------------------------------------------- native C++ library
FROM ubuntu:24.04 AS native-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends g++ cmake ninja-build ccache \
 && rm -rf /var/lib/apt/lists/*
ENV CCACHE_DIR=/ccache/native
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tests tests
COPY examples examples
RUN --mount=type=cache,target=/ccache/native \
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
 && cmake --build build \
 && ccache --show-stats \
 && ctest --test-dir build --output-on-failure \
 && cmake --install build --prefix /opt/syn

FROM ubuntu:24.04 AS native
COPY --from=native-build /opt/syn /opt/syn
COPY --from=native-build /src/build/learning_theories_demo /usr/local/bin/
COPY --from=native-build /src/build/syn_tests /usr/local/bin/
CMD ["learning_theories_demo"]

# ---------------------------------------------------------------- WebAssembly module
FROM emscripten/emsdk:${EMSDK_VERSION} AS wasm-build
RUN apt-get update \
 && apt-get install -y --no-install-recommends ccache \
 && rm -rf /var/lib/apt/lists/*
# emcc runs its clang through ccache (the bytes of syn.wasm do not change).
ENV CCACHE_DIR=/ccache/wasm EM_COMPILER_WRAPPER=ccache
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY wasm wasm
# Built from source into web/wasm: never the committed binaries.
RUN --mount=type=cache,target=/ccache/wasm \
    emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build-wasm -j"$(nproc)" \
 && ccache --show-stats

FROM wasm-build AS wasm
COPY web/js web/js
COPY web/tests web/tests
RUN node web/tests/wasm.test.mjs

FROM scratch AS wasm-artifacts
COPY --from=wasm /src/web/wasm/syn.js /src/web/wasm/syn.wasm /

# The development site: web/ with the module built from source in place of the committed one.
FROM scratch AS site
COPY web /web
COPY --from=wasm /src/web/wasm/syn.js /src/web/wasm/syn.wasm /web/wasm/

# ---------------------------------------------------------------- standalone (copy-deployable) site
# One folder of static files with the .wasm embedded: no ES modules, no fetch, no server
# configuration. Open index.html from disk or copy the folder to any static host.
FROM node:${NODE_VERSION}-alpine AS standalone
ARG GIT_COMMIT=unknown
WORKDIR /src
COPY tools/standalone/package.json tools/standalone/package-lock.json tools/standalone/
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi \
 && cd tools/standalone && npm ci --no-audit --no-fund
COPY tools/standalone/build.mjs tools/standalone/
COPY --from=site /web web
RUN GIT_COMMIT=${GIT_COMMIT} node tools/standalone/build.mjs --out /dist

FROM scratch AS standalone-artifacts
COPY --from=standalone /dist /

# ---------------------------------------------------------------- browser end-to-end tests
FROM mcr.microsoft.com/playwright:v${PLAYWRIGHT_VERSION}-noble AS e2e
ARG PLAYWRIGHT_VERSION
# Behind a TLS-intercepting proxy, pass its CA: docker build --secret id=ca,src=/path/ca.pem ...
WORKDIR /opt/e2e
RUN --mount=type=secret,id=ca,required=false \
    if [ -f /run/secrets/ca ]; then export NODE_EXTRA_CA_CERTS=/run/secrets/ca; fi \
 && npm init -y >/dev/null && npm install --no-audit --no-fund playwright@${PLAYWRIGHT_VERSION}
ENV PLAYWRIGHT_MODULE=/opt/e2e/node_modules/playwright/index.mjs
COPY --from=site /web /web
COPY --from=standalone /dist/standalone /standalone
WORKDIR /web
# Every model's kernel on WebGPU and on the software GPU; the development site over HTTP; then the
# standalone build opened from disk and served over HTTP.
RUN node tests/kernels.mjs \
 && node tests/e2e.mjs \
 && node tests/e2e.mjs --root /standalone --file \
 && node tests/e2e.mjs --root /standalone

# ---------------------------------------------------------------- web front end (default)
FROM nginx:1.27-alpine AS web
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=standalone /dist/standalone /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=3s CMD wget -q -O /dev/null http://127.0.0.1/ || exit 1
