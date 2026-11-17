# Dockerfile  [The API image]
# -------------------------------------------------------------------
# Purpose   One image that runs the API the same way on a laptop, in CI and on Render
# Build     docker build -t etf-advisor .
# Run       docker run --rm -p 8000:8000 -e MODEL_SOURCE=local etf-advisor
# Contents  conda-lock.yml, config.yaml and src/ only (.dockerignore is an allow-list); never .env or data
# Why       Exact package versions from conda-lock.yml: what passed CI is what runs in the cloud

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

# Logs reach the console at once (Docker and Render show them live); no .pyc files written at run time
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# 4. Runtime: the port, a health check with Python's own urllib (no curl in the image), and the start command
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["/usr/local/bin/_entrypoint.sh", "python", "-c", "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8000') + '/health', timeout=4)"]

# Render sets PORT; locally it falls back to 8000. exec makes uvicorn the main process, so it receives stop signals
CMD ["sh", "-c", "exec uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
