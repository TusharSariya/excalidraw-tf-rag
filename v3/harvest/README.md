# harvest

## Docker

Build:

```bash
docker build -t harvest:dev .
```

Run (secrets come from `.env`, gitignored — see `.env` for the expected keys):

```bash
docker run --rm -e OPENALEX_API_KEY --env-file .env harvest:dev --arxiv_id 1706.03762
```

## Logging

Logs go to stdout (captured by `docker logs`). The image sets `PYTHONUNBUFFERED=1` so
lines stream immediately instead of sitting in a buffer.

The log level is controlled by the `LOG_LEVEL` env var (default `INFO`; accepts
`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, case-insensitive):

```bash
docker run --rm -e LOG_LEVEL=DEBUG --env-file .env harvest:dev --arxiv_id 1706.03762
```

Loggers are namespaced under `harvest` (`harvest.arxiv`, `harvest.semantic_scholar`),
so you can filter a run by component:

```bash
docker logs <container> 2>&1 | grep harvest.semantic_scholar
```
