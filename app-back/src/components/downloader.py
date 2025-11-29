"""
YouTube downloader helper using yt_dlp.

Functions:
  download_youtube(url, out_dir) -> path to downloaded video file

This module uses `yt_dlp`. If it's not installed, the function will
raise an informative RuntimeError.
"""
from pathlib import Path
import shutil
import subprocess

def download_youtube(url: str, out_dir: str) -> str:
    """Download the provided YouTube URL into `out_dir` and return the file path.

    The function prefers the bestvideo+bestaudio merged format and saves as MP4.
    It requires `yt-dlp` to be installed and available in PATH.
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    ytdlp = shutil.which("yt-dlp") or shutil.which("yt_dl")
    if not ytdlp:
        raise RuntimeError("yt-dlp is not installed or not found in PATH. Please install yt-dlp to enable YouTube downloads.")

    # Template: save as numeric id to avoid weird characters
    out_template = str(out_dir_p / "%(id)s.%(ext)s")
    cmd = [ytdlp, "-f", "bestaudio[ext=m4a]+bestvideo[ext=mp4]/best", "-o", out_template, url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp failed: {result.stdout}")

    # Find the downloaded file (most recent file in out_dir)
    files = sorted(out_dir_p.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise RuntimeError("No file was downloaded by yt-dlp")
    return str(files[0])
