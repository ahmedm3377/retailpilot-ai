# The backend image: API, MCP server, adapter and admin commands (same code, same uv.lock as a native run).
# Build context is the repository root (see deploy/compose.yaml). Needs the internet once, at build time:
# Python packages, the spaCy model (from uv.lock) and the guardrail models are all baked in, so the
# containers run offline afterwards (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE are set by the app itself).
FROM ghcr.io/astral-sh/uv:0.9.24 AS uv

FROM python:3.13-slim-bookworm
COPY --from=uv /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

# Non-root user; the guardrail models land in its home caches (~/.cache/huggingface, ~/.cache/torch, ~/nltk_data).
RUN useradd --create-home --uid 1000 app && mkdir /app && chown app:app /app
USER app
WORKDIR /app

# Dependencies first (cached layer), then the project itself, installed editable like `uv sync` does natively,
# so paths such as the repository root (.env.example, candidate_assets/) resolve exactly as on the host.
COPY --chown=app:app pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=app:app . .
RUN uv sync --frozen --no-dev

# The local guardrail models (README "One-time setup"), downloaded once here, never at runtime.
RUN python -m guardrails_ai.detect_jailbreak.post_install \
 && python -m guardrails_ai.toxic_language.post_install

CMD ["python", "-m", "app.api.serve", "--no-start-stack"]
