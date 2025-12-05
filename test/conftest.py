import sys
import os
import importlib.util
from pathlib import Path
import pytest
import importlib


# Ensure test runtime dependencies are present with a clear message if not.
if importlib.util.find_spec("httpx") is None:
    pytest.exit(
        "Missing test dependency 'httpx'. Install test requirements with: `pip install -r requirements.txt`",
        returncode=2,
    )


# Make `app-back` importable for tests that do module-level imports.
# Some test modules import `src.components.*` at import time, so ensure the
# app-back directory is on `sys.path` immediately when conftest is imported.
current = Path(__file__).resolve()
repo_root = None
for p in current.parents:
    if (p / "app-back").is_dir():
        repo_root = p
        break
if repo_root is None:
    repo_root = current.parents[1]
app_back = repo_root / "app-back"
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
if str(app_back) not in sys.path:
    sys.path.insert(0, str(app_back))


@pytest.fixture(scope="session")
def app_module(tmp_path_factory):
    """Load the `app-back/main.py` module as `app_module` and ensure imports work.

    This fixture inserts the `app-back` directory on `sys.path` so the
    `src.components` imports resolve when the module is executed.
    """
    # Find the repository root by walking parents until we find `app-back`.
    # This is more robust than assuming a fixed number of parent levels
    # (the tests may be run from different working directories).
    current = Path(__file__).resolve()
    repo_root = None
    for p in current.parents:
        if (p / "app-back").is_dir():
            repo_root = p
            break
    if repo_root is None:
        # fallback to the original heuristic (sibling of `test`)
        repo_root = current.parents[1]
    app_back = repo_root / "app-back"
    sys.path.insert(0, str(app_back))

    spec = importlib.util.spec_from_file_location("app_main", str(app_back / "main.py"))
    app_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_main)

    yield app_main

    # cleanup: remove the path we inserted
    try:
        sys.path.remove(str(app_back))
    except Exception:
        pass


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient

    client = TestClient(app_module.app)
    return client


@pytest.fixture()
def tmp_output_dir(tmp_path, monkeypatch, app_module):
    # Point the app's BASE_DIR and OUTPUT_MP_DIR to an isolated tmp path
    base = tmp_path / "app_back"
    base.mkdir()
    monkeypatch.setattr(app_module, "BASE_DIR", str(base))
    # ensure the mp/output_mp exists
    mp_dir = base / "mp"
    (mp_dir / "output_mp").mkdir(parents=True, exist_ok=True)
    return base
