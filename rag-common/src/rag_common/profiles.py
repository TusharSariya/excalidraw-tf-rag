"""Named embedding profiles (TOML) with legacy env fallback."""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from rag_common.config import (
    EmbedBackend,
    EmbedConfig,
    local_config,
)

log = logging.getLogger("rag_common.profiles")

_BUILTIN_PROFILES_PATH = Path(__file__).with_name("embed_profiles.toml")

_VALID_BACKENDS = frozenset({"local", "auto"})


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    backend: str
    model: str
    dimensions: int
    quant: str | None = None

    def to_embed_config(self) -> EmbedConfig:
        backend = self._resolved_backend()
        return EmbedConfig(
            backend=backend,
            model=self.model,
            dimensions=self.dimensions,
            profile=self.name,
            quant=self.quant,
        )

    def _resolved_backend(self) -> EmbedBackend:
        if self.backend == "auto":
            raise ValueError(f"Profile {self.name!r} uses backend=auto; resolve before converting")
        if self.backend != "local":
            raise ValueError(f"Profile {self.name!r} has invalid backend {self.backend!r}")
        return self.backend  # type: ignore[return-value]


def _parse_profiles_table(data: dict[str, Any], *, source: Path) -> dict[str, ProfileSpec]:
    raw = data.get("profiles", data)
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid profiles file {source}: missing [profiles] table")
    out: dict[str, ProfileSpec] = {}
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        backend = str(spec.get("backend", "")).strip().lower()
        model = str(spec.get("model", "")).strip()
        if not backend or not model:
            raise ValueError(f"Profile {name!r} in {source} requires backend and model")
        dims = int(spec.get("dimensions", 0))
        if dims <= 0:
            raise ValueError(f"Profile {name!r} in {source} requires positive dimensions")
        quant_raw = spec.get("quant")
        quant = str(quant_raw).strip().lower() if quant_raw else None
        if quant in ("", "none", "off", "false", "0"):
            quant = None
        out[name] = ProfileSpec(
            name=name,
            backend=backend,
            model=model,
            dimensions=dims,
            quant=quant,
        )
    return out


def _load_profiles_file(path: Path) -> dict[str, ProfileSpec]:
    if not path.is_file():
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return _parse_profiles_table(data, source=path)


@lru_cache(maxsize=8)
def _merged_profiles_cached(merge_key: tuple[str, ...]) -> dict[str, ProfileSpec]:
    merged: dict[str, ProfileSpec] = {}
    for path_str in merge_key:
        for name, spec in _load_profiles_file(Path(path_str)).items():
            merged[name] = spec
    return merged


def load_profiles(*, extra_paths: list[Path] | None = None) -> dict[str, ProfileSpec]:
    """Load built-in profiles, then merge overrides (later paths win)."""
    builtin = _BUILTIN_PROFILES_PATH.resolve()
    tool_paths = [p.resolve() for p in (extra_paths or []) if p.resolve() != builtin]
    env_raw = os.getenv("RAG_EMBED_PROFILES_PATH")
    env_path = Path(env_raw).resolve() if env_raw else None

    merge_order: list[Path] = [builtin, *tool_paths]
    if env_path and env_path.is_file():
        merge_order.append(env_path)

    return _merged_profiles_cached(tuple(str(p) for p in merge_order))


def list_profile_names(*, extra_paths: list[Path] | None = None) -> list[str]:
    return sorted(load_profiles(extra_paths=extra_paths).keys())


def get_profile(name: str, *, extra_paths: list[Path] | None = None) -> ProfileSpec:
    profiles = load_profiles(extra_paths=extra_paths)
    if name not in profiles:
        available = ", ".join(sorted(profiles)) or "(none)"
        raise KeyError(f"Unknown embed profile {name!r}. Available: {available}")
    return profiles[name]


def _legacy_default_profile_name() -> str:
    return "default"


def _config_from_legacy_env(prefix: str) -> EmbedConfig:
    """Synthesize embed config from RAG_EMBED_BACKEND / model env vars (pre-profile behavior)."""
    mode = (
        os.getenv(f"{prefix}EMBED_BACKEND", os.getenv("RAG_EMBED_BACKEND", "auto")).strip().lower()
    )

    if mode not in ("local", "auto"):
        raise RuntimeError(f"Unknown embed backend mode: {mode!r} (use auto or local)")

    cfg = local_config()
    quant = os.getenv("RAG_LOCAL_EMBED_QUANT", "").strip().lower() or None
    if quant in ("", "none", "off", "false", "0"):
        quant = None
    return EmbedConfig(
        backend=cfg.backend,
        model=cfg.model,
        dimensions=cfg.dimensions,
        profile=_legacy_default_profile_name(),
        quant=quant,
    )


def resolve_profile_name(
    *,
    prefix: str = "",
    profile: str | None = None,
) -> str | None:
    if profile:
        return profile.strip()
    for key in (f"{prefix}EMBED_PROFILE", "RAG_EMBED_PROFILE"):
        value = os.getenv(key)
        if value:
            return value.strip()
    return None


def resolve_embed_profile_config(
    *,
    prefix: str = "",
    profile: str | None = None,
    extra_paths: list[Path] | None = None,
) -> EmbedConfig:
    name = resolve_profile_name(prefix=prefix, profile=profile)
    if not name or name == _legacy_default_profile_name():
        return _config_from_legacy_env(prefix)

    spec = get_profile(name, extra_paths=extra_paths)
    cfg = spec.to_embed_config()
    log.info(
        "embed profile=%s backend=%s model=%s dims=%d quant=%s",
        cfg.profile,
        cfg.backend,
        cfg.model,
        cfg.dimensions,
        cfg.quant or "-",
    )
    return cfg
