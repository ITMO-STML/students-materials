FROM ghcr.io/astral-sh/uv:python3.12-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

COPY . /app

ENV UV_NO_DEV=1

WORKDIR /app
RUN uv sync --locked --no-cache
