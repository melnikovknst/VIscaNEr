"""Configuration loading and path resolution for the bottle reranker pipeline.

The config is a plain nested mapping. Access goes through :class:`Config`, which
resolves every path against ``project_root`` and refuses silently-missing keys,
so a typo in the YAML fails at load instead of halfway through a 45k run.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:  # PyYAML is in requirements.txt; JSON configs work without it.
    import yaml
except ImportError:  # pragma: no cover - exercised only on bare interpreters
    yaml = None

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "bottle_reranker.yaml"

_MISSING = object()


class ConfigError(RuntimeError):
    """Raised for a malformed or incomplete configuration."""


@dataclass
class SourceSpec:
    """One pool of query images."""

    name: str
    kind: str
    photographic: bool
    enabled: bool
    images_root: Path
    manifest_csv: Path
    path_column: str
    slug_column: str
    group_columns: list[str] = field(default_factory=list)
    exact_duplicate_column: str | None = None
    require_columns: dict[str, str] = field(default_factory=dict)
    bottle_manifest_csv: Path | None = None

    @property
    def is_synthetic(self) -> bool:
        return not self.photographic


@dataclass
class Config:
    raw: dict[str, Any]
    project_root: Path
    config_path: Path | None = None

    # ------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None, *, project_root: str | os.PathLike[str] | None = None) -> "Config":
        config_path = Path(path) if path else DEFAULT_CONFIG
        if not config_path.is_file():
            raise ConfigError(f"Config file not found: {config_path}")
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise ConfigError(
                    "PyYAML is required to read a YAML config. "
                    "Install it (pip install PyYAML) or pass a .json config."
                )
            raw = yaml.safe_load(text)
        else:
            import json

            raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ConfigError(f"Config root must be a mapping, got {type(raw).__name__}")

        # An explicit override is relative to the caller's working directory;
        # the value in the file is relative to the file, so a config can be
        # moved around with its paths intact.
        if project_root:
            root = Path(project_root).expanduser().resolve()
        else:
            root = Path(raw.get("project_root") or ".").expanduser()
            if not root.is_absolute():
                root = (config_path.parent / root).resolve()
        if not root.is_dir():
            raise ConfigError(f"project_root does not exist: {root}")
        return cls(raw=raw, project_root=root, config_path=config_path)

    # ------------------------------------------------------------- lookups
    def get(self, dotted: str, default: Any = _MISSING) -> Any:
        """Fetch ``a.b.c``. Without a default a missing key is an error."""
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is _MISSING:
                    raise ConfigError(f"Missing config key: {dotted}")
                return default
            node = node[part]
        return copy.deepcopy(node) if isinstance(node, (dict, list)) else node

    def path(self, dotted: str, default: Any = _MISSING) -> Path:
        """Fetch a key and resolve it against ``project_root``."""
        value = self.get(dotted, default)
        return self.resolve(value)

    def resolve(self, value: str | os.PathLike[str]) -> Path:
        candidate = Path(value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.project_root / candidate).resolve()

    def relative(self, value: str | os.PathLike[str]) -> str:
        """Render a path relative to ``project_root`` when it lives inside it."""
        candidate = Path(value).resolve()
        try:
            return candidate.relative_to(self.project_root).as_posix()
        except ValueError:
            return candidate.as_posix()

    # ------------------------------------------------------------- sources
    def sources(self, *, enabled_only: bool = True) -> list[SourceSpec]:
        specs: list[SourceSpec] = []
        entries = self.get("sources")
        if not isinstance(entries, list) or not entries:
            raise ConfigError("`sources` must be a non-empty list")
        seen: set[str] = set()
        for entry in entries:
            required = {"name", "kind", "photographic", "images_root", "manifest_csv",
                        "path_column", "slug_column"}
            missing = sorted(required - set(entry))
            if missing:
                raise ConfigError(f"Source {entry.get('name', '?')} is missing keys: {missing}")
            if entry["name"] in seen:
                raise ConfigError(f"Duplicate source name: {entry['name']}")
            seen.add(entry["name"])
            spec = SourceSpec(
                name=str(entry["name"]),
                kind=str(entry["kind"]),
                photographic=bool(entry["photographic"]),
                enabled=bool(entry.get("enabled", True)),
                images_root=self.resolve(entry["images_root"]),
                manifest_csv=self.resolve(entry["manifest_csv"]),
                path_column=str(entry["path_column"]),
                slug_column=str(entry["slug_column"]),
                group_columns=[str(c) for c in entry.get("group_columns", [])],
                exact_duplicate_column=entry.get("exact_duplicate_column"),
                require_columns={str(k): str(v) for k, v in (entry.get("require_columns") or {}).items()},
                bottle_manifest_csv=(
                    self.resolve(entry["bottle_manifest_csv"])
                    if entry.get("bottle_manifest_csv")
                    else None
                ),
            )
            if enabled_only and not spec.enabled:
                continue
            specs.append(spec)
        if not specs:
            raise ConfigError("No enabled sources in the config")
        return specs

    def source(self, name: str) -> SourceSpec:
        for spec in self.sources(enabled_only=False):
            if spec.name == name:
                return spec
        raise ConfigError(f"Unknown source: {name}")

    # ------------------------------------------------------------- outputs
    @property
    def output_root(self) -> Path:
        return self.path("output.root")

    @property
    def reports_root(self) -> Path:
        return self.path("output.reports_root")

    def output_dir(self, *parts: str, create: bool = True) -> Path:
        target = self.output_root.joinpath(*parts)
        if create:
            target.mkdir(parents=True, exist_ok=True)
        return target

    def report_path(self, name: str, *, create_parent: bool = True) -> Path:
        target = self.reports_root / name
        if create_parent:
            target.parent.mkdir(parents=True, exist_ok=True)
        return target

    # --------------------------------------------------------------- misc
    def fingerprint(self, keys: Iterable[str]) -> dict[str, Any]:
        """A small dict of the settings that determine a stage result.

        Written into every stage record so a rebuild with different settings is
        visible instead of silently mixing outputs.
        """
        return {key: self.get(key, None) for key in keys}
