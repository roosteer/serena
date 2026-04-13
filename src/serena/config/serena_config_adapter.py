"""
Local adapter for :class:`SerenaConfig` that adds environment variable expansion
in project path loading and template preservation during save.

This file is intentionally kept separate from the upstream ``serena_config.py`` to
minimise rebase conflicts when syncing with oraios/serena. The upstream entry points
(``agent.py``, ``cli.py``, ``mcp.py``, ``project_server.py``) shadow the
``SerenaConfig`` name with the adapter via an import alias::

    from serena.config.serena_config_adapter import SerenaConfigAdapter as SerenaConfig
"""

import os
import re
from pathlib import Path
from typing import Optional

from sensai.util import logging

from serena.config.serena_config import (
    ProjectConfig,
    RegisteredProject,
    SerenaConfig,
    SerenaConfigError,
)
from serena.util.yaml import load_yaml

log = logging.getLogger(__name__)

# matches $VAR and ${VAR} patterns in shell-style template strings
SHELL_VARIABLE_PATTERN = re.compile(r"\$\{([A-Za-z_]\w*)\}|\$([A-Za-z_]\w*)")


class SerenaConfigAdapter(SerenaConfig):
    """
    Subclass of :class:`SerenaConfig` that adds environment variable expansion
    (``$VAR`` / ``${VAR}`` / ``%VAR%``) in project path entries and preserves
    the original template strings when the configuration is saved back to disk.

    The adapter is designed to be a drop-in replacement: all call sites that
    reference ``SerenaConfig`` by name can use this class transparently via an
    import alias.
    """

    # -------------------------------------------------------------------------
    # environment variable resolution helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _resolve_environment_variable(name: str) -> Optional[str]:
        """
        Looks up an environment variable by name.
        Treats ``home`` / ``HOME`` as ``Path.home()``.
        On Windows, falls back to a case-insensitive search.

        :param name: variable name without the leading ``$``
        :return: resolved string value, or ``None`` if not found
        """
        if name in {"home", "HOME"}:
            return str(Path.home())
        if name in os.environ:
            return os.environ[name]
        if os.name == "nt":
            lowered = name.lower()
            for key, value in os.environ.items():
                if key.lower() == lowered:
                    return value
        return None

    @classmethod
    def _resolve_template_variable(cls, name: str, placeholders: dict[str, str]) -> str:
        """
        Resolves a single template variable against *placeholders* first,
        then against the environment.

        :param name: variable name without the leading ``$``
        :param placeholders: mapping from name to replacement value
        :return: resolved replacement string
        :raises SerenaConfigError: if the variable is unknown in both sources
        """
        if name in placeholders:
            return placeholders[name]

        env_value = cls._resolve_environment_variable(name)
        if env_value is not None:
            return env_value

        raise SerenaConfigError(
            f"Unknown placeholder '${name}'. "
            f"Supported placeholders: {', '.join('$' + k for k in placeholders)} "
            "or any existing environment variable."
        )

    @classmethod
    def _expand_template_variables(cls, template: str, placeholders: dict[str, str]) -> str:
        """
        Expands ``$VAR`` and ``${VAR}`` tokens using *placeholders* and the
        environment, then applies :func:`os.path.expandvars` and
        :func:`os.path.expanduser` for any remaining shell substitutions.

        :param template: path template string, e.g. ``"$HOME/projects/foo"``
        :param placeholders: additional name-to-value substitutions
        :return: fully expanded path string
        :raises SerenaConfigError: if the template contains an unresolved Windows
            ``%VAR%`` reference after expansion
        """

        def _replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            assert name is not None
            return cls._resolve_template_variable(name, placeholders)

        expanded = SHELL_VARIABLE_PATTERN.sub(_replace, template)
        expanded = os.path.expanduser(os.path.expandvars(expanded))

        # guard against unresolved Windows-style %VAR% references
        if re.search(r"%[^%]+%", expanded):
            raise SerenaConfigError(
                f"Unresolved Windows environment variable in path: '{template}'"
            )
        return expanded

    # -------------------------------------------------------------------------
    # override: folder location resolution (now supports env vars)
    # -------------------------------------------------------------------------

    @classmethod
    def _resolve_serena_folder_location(cls, template: str, placeholders: dict[str, str]) -> str:
        """
        Resolves a folder location template by expanding ``$placeholder`` tokens,
        environment variables (``$VAR`` / ``${VAR}``), and ``~``.

        :param template: the template string (e.g. ``"$projectDir/.serena"``)
        :param placeholders: mapping from placeholder name to replacement value
        :return: the resolved absolute path
        :raises SerenaConfigError: if the template contains an unrecognised token
        """
        result = cls._expand_template_variables(template, placeholders)
        return os.path.abspath(result)

    # -------------------------------------------------------------------------
    # override: load projects with env var expansion and track templates
    # -------------------------------------------------------------------------

    @classmethod
    def from_config_file(cls, generate_if_missing: bool = True) -> "SerenaConfigAdapter":
        """
        Loads configuration from the standard file path.

        Extends the base implementation to:
        - expand ``$VAR`` / ``${VAR}`` tokens in project path entries;
        - load projects whose paths were skipped by the base loader because they
          contained unexpanded variables;
        - build a ``_template_map`` (resolved path → original template string) on the
          returned instance, used by :meth:`save` to round-trip template paths.

        :param generate_if_missing: whether to auto-generate the config file if absent
        :return: fully configured :class:`SerenaConfigAdapter` instance
        """
        # delegate the bulk of loading (file generation, YAML parse, migrations,
        # all non-project fields, and projects whose paths need no expansion)
        instance = super().from_config_file(generate_if_missing)

        # re-read the raw YAML to retrieve the original (possibly templated) path strings;
        # the base loader already resolved the file path, so we can reuse it
        assert instance.config_file_path is not None
        raw_yaml = load_yaml(instance.config_file_path)

        # build template map and recover skipped projects
        template_map: dict[str, str] = {}
        for raw_entry in raw_yaml.get("projects", []):
            raw_path_str = str(raw_entry)
            try:
                expanded = cls._expand_template_variables(raw_path_str, {})
            except SerenaConfigError as e:
                log.warning("Skipping project path '%s': %s", raw_path_str, e)
                continue

            resolved = Path(expanded).resolve()
            resolved_str = str(resolved)
            template_map[resolved_str] = raw_path_str

            # load projects that the base loader skipped (path had unexpanded vars)
            if not any(p.project_root == resolved for p in instance.projects):
                project_yml = instance.get_project_yml_location(resolved_str)
                if resolved.is_dir() and os.path.isfile(project_yml):
                    try:
                        project_config = ProjectConfig.load(resolved, serena_config=instance)
                        instance.projects.append(
                            RegisteredProject(
                                project_root=resolved_str,
                                project_config=project_config,
                            )
                        )
                        log.info("Loaded project from env-var path '%s'", raw_path_str)
                    except Exception as e:
                        log.warning(
                            "Could not load project at '%s' (template: '%s'): %s",
                            resolved_str,
                            raw_path_str,
                            e,
                        )

        instance._template_map = template_map
        return instance

    # -------------------------------------------------------------------------
    # override: preserve template strings when saving
    # -------------------------------------------------------------------------

    def save(self) -> None:
        """
        Saves the configuration to disk, preserving original template strings
        (e.g. ``$HOME/myproject``) in the ``projects`` list instead of writing
        the resolved absolute paths.

        Temporarily swaps ``project.project_root`` on each project whose resolved
        path has a known template, delegates to :meth:`SerenaConfig.save`, then
        restores the original values.
        """
        template_map = getattr(self, "_template_map", {})
        swapped: dict[RegisteredProject, Path] = {}

        for project in self.projects:
            template = template_map.get(str(project.project_root))
            if template is not None and template != str(project.project_root):
                swapped[project] = project.project_root
                project.project_root = Path(template)

        try:
            super().save()
        finally:
            for project, original in swapped.items():
                project.project_root = original
