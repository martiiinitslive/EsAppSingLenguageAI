import pytest
import importlib.util
from pathlib import Path
import shutil

repo = Path(__file__).resolve().parent.parent


def _load_renderer():
    mod_path = repo / 'app-back' / 'mp' / 'run_pose_to_video_mediapipe.py'
    spec = importlib.util.spec_from_file_location('rpp', str(mod_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_to_pose_sequence_basic():
    mod = _load_renderer()
    # Basic behavior: letters yield themselves, spaces -> SPACE, '.' -> PERIOD, ',' -> COMMA
    seq = mod.text_to_pose_sequence('Hi, ok.')
    assert 'SPACE' in seq or 'COMMA' in seq
    assert seq[-1] == 'PERIOD'
