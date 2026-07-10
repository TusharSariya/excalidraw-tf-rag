FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev
COPY src ./src
RUN uv sync --frozen --no-dev

FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim@sha256:7cf77f594be8042dab6daa9fe326f90962252268b4f120a7f5dccce4d947e6c1 AS final
WORKDIR /app
RUN useradd --create-home --uid 1000 --shell /usr/sbin/nologin harvest \
    && chown -R harvest:harvest /app
COPY --from=builder --chown=harvest:harvest /app/.venv /app/.venv
COPY --from=builder --chown=harvest:harvest /app/src /app/src
USER harvest
ENV PATH="/app/.venv/bin:$PATH" HOME=/home/harvest PYTHONUNBUFFERED=1
ENTRYPOINT ["harvest"]
