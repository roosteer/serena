"""
Tests for :class:`SerenaConfigAdapter`.

Coverage:
- environment variable resolution helpers
- template variable expansion
- _resolve_serena_folder_location override (now accepts env vars)
- from_config_file: template map building and round-trip of env-var paths
- save: template strings are preserved instead of resolved absolute paths
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from serena.config.serena_config import SerenaConfigError
from serena.config.serena_config_adapter import SerenaConfigAdapter
from serena.constants import SERENA_CONFIG_TEMPLATE_FILE
from serena.util.yaml import load_yaml, save_yaml


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_serena_home(tmp_dir: str, project_paths: list[str]) -> str:
    """
    Creates a minimal SERENA_HOME inside *tmp_dir* with a ``serena_config.yml``
    that lists the given *project_paths*.

    :param tmp_dir: parent temporary directory
    :param project_paths: raw path strings to write into the ``projects`` field
    :return: path to the created SERENA_HOME directory
    """
    serena_home = Path(tmp_dir) / "serena_home"
    serena_home.mkdir()

    # start from the official template so all required keys are present
    config_yaml = load_yaml(SERENA_CONFIG_TEMPLATE_FILE)
    config_yaml["projects"] = project_paths

    save_yaml(str(serena_home / "serena_config.yml"), config_yaml)
    return str(serena_home)


def _make_project_dir(base: str, name: str) -> Path:
    """
    Creates a minimal project directory (folder + ``.serena/project.yml``) under *base*.

    :param base: root temporary directory
    :param name: project sub-directory name
    :return: absolute path to the created project directory
    """
    project_root = Path(base) / name
    project_root.mkdir(parents=True)
    serena_dir = project_root / ".serena"
    serena_dir.mkdir()
    # minimal project.yml (content is not important for these tests)
    (serena_dir / "project.yml").write_text(
        "project_name: test\nlanguages: []\n", encoding="utf-8"
    )
    return project_root


# ---------------------------------------------------------------------------
# _resolve_environment_variable
# ---------------------------------------------------------------------------

class TestResolveEnvironmentVariable:
    """Unit tests for the env var lookup helper."""

    def test_home_alias_returns_home_dir(self):
        result = SerenaConfigAdapter._resolve_environment_variable("HOME")
        assert result == str(Path.home())

    def test_home_lowercase_alias_returns_home_dir(self):
        result = SerenaConfigAdapter._resolve_environment_variable("home")
        assert result == str(Path.home())

    def test_known_env_var_is_returned(self):
        with patch.dict(os.environ, {"MY_TEST_VAR": "/some/path"}):
            result = SerenaConfigAdapter._resolve_environment_variable("MY_TEST_VAR")
        assert result == "/some/path"

    def test_unknown_var_returns_none(self):
        # use a name very unlikely to be set in the real environment
        result = SerenaConfigAdapter._resolve_environment_variable("_SERENA_ADAPTER_NEVER_SET_XYZ_")
        assert result is None


# ---------------------------------------------------------------------------
# _expand_template_variables
# ---------------------------------------------------------------------------

class TestExpandTemplateVariables:
    """Unit tests for shell-style template expansion."""

    def test_dollar_home_is_expanded(self):
        result = SerenaConfigAdapter._expand_template_variables("$HOME/projects/foo", {})
        assert result == str(Path.home() / "projects" / "foo")

    def test_braced_home_is_expanded(self):
        result = SerenaConfigAdapter._expand_template_variables("${HOME}/projects/bar", {})
        assert result == str(Path.home() / "projects" / "bar")

    def test_tilde_is_expanded(self):
        result = SerenaConfigAdapter._expand_template_variables("~/projects/baz", {})
        assert result.startswith(str(Path.home()))

    def test_placeholder_is_resolved_before_env(self):
        """Values in the explicit placeholders dict take precedence over env vars."""
        with patch.dict(os.environ, {"MY_VAR": "from_env"}):
            result = SerenaConfigAdapter._expand_template_variables(
                "$MY_VAR/sub", {"MY_VAR": "from_placeholder"}
            )
        assert result == "from_placeholder/sub"

    def test_env_var_is_resolved_when_not_in_placeholders(self):
        with patch.dict(os.environ, {"MY_PROJECT_BASE": "/custom/base"}):
            result = SerenaConfigAdapter._expand_template_variables(
                "$MY_PROJECT_BASE/myproject", {}
            )
        assert result == "/custom/base/myproject"

    def test_unknown_var_raises_serena_config_error(self):
        with pytest.raises(SerenaConfigError, match="Unknown placeholder"):
            SerenaConfigAdapter._expand_template_variables(
                "$_UNDEFINED_VAR_XYZ_/path", {}
            )

    def test_plain_path_is_returned_unchanged(self):
        result = SerenaConfigAdapter._expand_template_variables("/absolute/path", {})
        assert result == "/absolute/path"


# ---------------------------------------------------------------------------
# _resolve_serena_folder_location (override)
# ---------------------------------------------------------------------------

class TestResolveSerenaFolderLocation:
    """Verify the adapter accepts env vars where the base class would raise."""

    def test_env_var_in_template_is_resolved(self):
        with patch.dict(os.environ, {"MY_DATA_DIR": "/data"}):
            result = SerenaConfigAdapter._resolve_serena_folder_location(
                "$MY_DATA_DIR/$projectFolderName/.serena",
                {"projectFolderName": "myproject"},
            )
        assert result == os.path.abspath("/data/myproject/.serena")

    def test_unknown_var_still_raises(self):
        with pytest.raises(SerenaConfigError):
            SerenaConfigAdapter._resolve_serena_folder_location(
                "$_UNKNOWN_ADAPTER_VAR_XYZ_/.serena", {}
            )

    def test_standard_placeholders_still_work(self):
        result = SerenaConfigAdapter._resolve_serena_folder_location(
            "$projectDir/.serena",
            {"projectDir": "/home/user/myproject"},
        )
        assert result == os.path.abspath("/home/user/myproject/.serena")


# ---------------------------------------------------------------------------
# from_config_file — template map and project loading
# ---------------------------------------------------------------------------

class TestFromConfigFileTemplateMap:
    """
    Tests that verify the adapter correctly builds *_template_map* and loads
    projects whose paths contained env var tokens.
    """

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_template_map_populated_for_literal_path(self):
        """
        A plain (non-template) path should map to itself in _template_map.
        """
        project_dir = _make_project_dir(self.tmp_dir, "myproject")
        serena_home = _make_serena_home(self.tmp_dir, [str(project_dir)])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home}):
            config = SerenaConfigAdapter.from_config_file()

        resolved_str = str(project_dir.resolve())
        assert resolved_str in config._template_map
        assert config._template_map[resolved_str] == str(project_dir)

    def test_project_with_env_var_path_is_loaded(self):
        """
        A project path written as ``$MY_BASE/myproject`` in the config should be
        loaded after env-var expansion.
        """
        project_dir = _make_project_dir(self.tmp_dir, "envproject")
        base_dir = str(project_dir.parent)

        serena_home = _make_serena_home(self.tmp_dir, ["$_SERENA_TEST_BASE_/envproject"])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home, "_SERENA_TEST_BASE_": base_dir}):
            config = SerenaConfigAdapter.from_config_file()

        loaded_roots = [str(p.project_root) for p in config.projects]
        assert str(project_dir.resolve()) in loaded_roots

    def test_template_map_preserves_env_var_string(self):
        """
        When the config has ``$MY_BASE/proj``, *_template_map* must keep the
        unexpanded template string — not the resolved path.
        """
        project_dir = _make_project_dir(self.tmp_dir, "tplproject")
        base_dir = str(project_dir.parent)
        raw_template = "$_SERENA_TEST_BASE2_/tplproject"

        serena_home = _make_serena_home(self.tmp_dir, [raw_template])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home, "_SERENA_TEST_BASE2_": base_dir}):
            config = SerenaConfigAdapter.from_config_file()

        resolved_str = str(project_dir.resolve())
        assert config._template_map.get(resolved_str) == raw_template

    def test_unknown_env_var_in_path_is_skipped_gracefully(self):
        """
        A project path containing an unresolvable variable should be silently
        skipped — no exception propagated to the caller.
        """
        serena_home = _make_serena_home(
            self.tmp_dir, ["$_SERENA_UNDEFINED_VAR_XYZ_/myproject"]
        )

        with patch.dict(os.environ, {"SERENA_HOME": serena_home}):
            config = SerenaConfigAdapter.from_config_file()  # must not raise

        assert config.projects == []


# ---------------------------------------------------------------------------
# save — template preservation
# ---------------------------------------------------------------------------

class TestSaveTemplatePreservation:
    """
    Verify that when the adapter saves the configuration, original env-var
    template strings are written back instead of the resolved absolute paths.
    """

    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmp_dir)

    def test_save_preserves_env_var_template(self):
        project_dir = _make_project_dir(self.tmp_dir, "saveproject")
        base_dir = str(project_dir.parent)
        raw_template = "$_SERENA_SAVE_BASE_/saveproject"

        serena_home = _make_serena_home(self.tmp_dir, [raw_template])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home, "_SERENA_SAVE_BASE_": base_dir}):
            config = SerenaConfigAdapter.from_config_file()
            config.save()

            reloaded_yaml = load_yaml(config.config_file_path)

        assert raw_template in reloaded_yaml["projects"]

    def test_save_does_not_write_resolved_path_when_template_exists(self):
        project_dir = _make_project_dir(self.tmp_dir, "resolvedproject")
        base_dir = str(project_dir.parent)
        raw_template = "$_SERENA_RESOLVED_BASE_/resolvedproject"
        resolved_path = str(project_dir.resolve())

        serena_home = _make_serena_home(self.tmp_dir, [raw_template])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home, "_SERENA_RESOLVED_BASE_": base_dir}):
            config = SerenaConfigAdapter.from_config_file()
            config.save()

            reloaded_yaml = load_yaml(config.config_file_path)

        # the resolved absolute path must NOT appear; only the template should
        assert resolved_path not in reloaded_yaml["projects"]
        assert raw_template in reloaded_yaml["projects"]

    def test_save_with_no_template_map_falls_back_to_resolved_paths(self):
        """
        If _template_map is absent (e.g. adapter constructed directly, not via
        from_config_file), save delegates cleanly to the base class behaviour.
        """
        project_dir = _make_project_dir(self.tmp_dir, "directproject")
        serena_home = _make_serena_home(self.tmp_dir, [str(project_dir)])

        with patch.dict(os.environ, {"SERENA_HOME": serena_home}):
            config = SerenaConfigAdapter.from_config_file()
            # remove _template_map to simulate direct construction
            if hasattr(config, "_template_map"):
                del config._template_map
            config.save()  # must not raise

            reloaded_yaml = load_yaml(config.config_file_path)

        assert len(reloaded_yaml["projects"]) == 1
