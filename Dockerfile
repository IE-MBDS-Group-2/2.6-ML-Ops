# Dockerfile  [The API image]
# -------------------------------------------------------------------
# Purpose   One image that runs the API the same way on a laptop, in CI and on the host
# Build     docker build -t etf-advisor .     (deploy.yml adds --build-arg GIT_COMMIT=<sha> APP_VERSION=<tag>)
# Run       docker run --rm -p 8000:8000 -e MODEL_SOURCE=local etf-advisor
# Contents  conda-lock.yml, config.yaml and src/ only (.dockerignore is an allow-list); never .env or data
# Why       Exact package versions from conda-lock.yml, and the image CI built is the image the host runs (GHCR)

# 1. Base: micromamba (a small conda) at a pinned version, on the platform the lock file was made for (linux-64)
FROM --platform=linux/amd64 mambaorg/micromamba:2.10.0

WORKDIR /app

# 2. Packages: copied and installed before the code, so this slow layer is rebuilt only when the lock changes
COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /app/conda-lock.yml
RUN micromamba install --yes --name base --file /app/conda-lock.yml \
    && micromamba clean --all --yes

# 3. Code and settings (secrets come from the host's environment variables, never from a file in the image)
COPY --chown=$MAMBA_USER:$MAMBA_USER config.yaml /app/config.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER src/ /app/src/

# Logs reach the console at once (docker logs shows them live); no .pyc files written at run time
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Which build this is, baked in: the host follows a moving tag, so it cannot know the commit or the release name.
# /health reports both (version, commit); the deploy's smoke test waits for this commit. The labels override the
# base image's own (it calls itself "latest").
ARG GIT_COMMIT=local
ARG APP_VERSION=dev
ENV RENDER_GIT_COMMIT=$GIT_COMMIT \
    APP_VERSION=$APP_VERSION
LABEL org.opencontainers.image.revision=$GIT_COMMIT \
      org.opencontainers.image.version=$APP_VERSION \
      org.opencontainers.image.source="https://github.com/IE-MBDS-Group-2/2.6-ML-Ops" \
      org.opencontainers.image.title="ETF rebalancing advisor API"

# 4. Runtime: the port, a health check with Python's own urllib (no curl in the image), and the start command
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/_entrypoint.sh", "python", "-c", "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"]

# A host may set PORT; otherwise 8000. exec makes uvicorn the main process, so it receives stop signals
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
