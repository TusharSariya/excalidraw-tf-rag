#!/usr/bin/env bash
# Actually executes changed GH Actions workflows locally via `act`, before
# they ever reach a real push — catches logic bugs (wrong CLI flags, silent
# no-ops) that actionlint's static checks can't see. Only runs when a
# workflow file is part of what's being pushed (see the `files:` filter on
# this hook in .pre-commit-config.yaml) — everyday code pushes stay fast,
# per this file's own "seconds, no tests" convention; the act cost is paid
# only when you're actually editing a workflow.
set -euo pipefail

if ! command -v act >/dev/null 2>&1; then
    echo "ERROR: 'act' is not installed but a workflow file changed."
    echo "Install it once: brew install act"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo "ERROR: Docker isn't running — 'act' needs it to execute workflows locally."
    exit 1
fi

# act defaults to /var/run/docker.sock, which isn't where every Docker
# provider (OrbStack, Colima, Docker Desktop with a custom context, ...)
# actually listens. Point it at whatever the active docker context uses,
# rather than assuming the default.
if [ -z "${DOCKER_HOST:-}" ]; then
    DETECTED_HOST=$(docker context inspect --format '{{.Endpoints.docker.Host}}' 2>/dev/null || true)
    [ -n "$DETECTED_HOST" ] && export DOCKER_HOST="$DETECTED_HOST"
fi

EVENT_FILE=".github/act-event-pull_request.json"
fail=0

for wf in "$@"; do
    echo "--- act pull_request -W $wf ---"
    # db-migrations.yml's validate job runs `supabase start` (Docker-in-Docker)
    # — nested Docker under act can be flaky/slow independent of whether the
    # workflow logic is actually broken. A failure here is still surfaced
    # (the user explicitly accepted this tradeoff), but the note below helps
    # tell "real bug" apart from "nested Docker flakiness."
    if [[ "$wf" == *db-migrations.yml ]]; then
        echo "note: this workflow runs Docker-in-Docker (supabase start) — a failure"
        echo "here may be local nested-Docker flakiness, not a real bug. Re-run once"
        echo "before assuming the workflow itself is broken."
    fi
    # -P pins the runner image non-interactively (act otherwise prompts on
    # first run and hangs any non-interactive caller, like a pre-push hook).
    # Medium image: small enough to be practical, compatible with the
    # standard actions this repo uses (checkout, setup-cli, buildx, etc.).
    if ! act pull_request -W "$wf" -e "$EVENT_FILE" --pull=false \
        -P ubuntu-latest=catthehacker/ubuntu:act-latest; then
        echo "ERROR: act failed on $wf"
        echo "note: matrix jobs run concurrently as local containers sharing the same"
        echo "cache dirs (e.g. setup-uv's cache) — unlike GitHub's isolated hosted"
        echo "runners, this can race and fail a job that's actually fine. Before"
        echo "assuming a real bug, re-run just that job in isolation:"
        echo "  act pull_request -W \"$wf\" -e \"$EVENT_FILE\" -P ubuntu-latest=catthehacker/ubuntu:act-latest -j <job-name>"
        fail=1
    fi
done

exit $fail
