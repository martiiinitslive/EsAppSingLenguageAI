"""
Launcher script to start backend and frontend for local development.

Usage:
  python launher.py

Behavior:
  - Uses `.venv\Scripts\python.exe` if present to start the backend `app-back/run_api.py`.
  - Runs `npm install` (if needed) and `npm start` inside `app-front` to start the frontend.
  - Opens the frontend URL in the default browser (http://localhost:3000) by default.
  - Runs both processes concurrently, shows their PIDs and forwards Ctrl-C to stop both.

Note: This script is intended for Windows development environment (PowerShell/CMD).
"""
from pathlib import Path
import subprocess
import sys
import os
import time
import shutil
import webbrowser


def find_venv_python(root: Path) -> str:
    """Return the path to the venv python.exe if available, otherwise sys.executable."""
    venv_py = root / ".venv" / "Scripts" / "python.exe"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def run_backend(root: Path, python_exe: str, new_console: bool = True):
    backend_script = root / "app-back" / "run_api.py"
    if not backend_script.exists():
        raise FileNotFoundError(f"Backend entry script not found: {backend_script}")

    cmd = [python_exe, str(backend_script)]
    print(f"Starting backend: {' '.join(cmd)}")
    kwargs = {"cwd": str(root / "app-back")}
    if os.name == "nt" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    proc = subprocess.Popen(cmd, **kwargs)
    print(f"Backend started with PID {proc.pid}")
    return proc


def run_frontend(root: Path, new_console: bool = True):
    front_dir = root / "app-front"
    if not front_dir.exists():
        raise FileNotFoundError(f"Frontend folder not found: {front_dir}")

    npm = shutil.which("npm")
    if npm is None:
        # On Windows try npm.cmd fallback
        npm = shutil.which("npm.cmd") or shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm not found in PATH. Install Node.js and npm to run the frontend.")

    # Ensure node modules installed
    node_modules = front_dir / "node_modules"
    if not node_modules.exists():
        print("node_modules not found, running 'npm install'... (this may take a while)")
        install_cmd = [npm, "install"]
        subprocess.check_call(install_cmd, cwd=str(front_dir))

    start_cmd = [npm, "start"]
    print(f"Starting frontend: {' '.join(start_cmd)} in {front_dir}")
    kwargs = {"cwd": str(front_dir)}
    if os.name == "nt" and new_console:
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    proc = subprocess.Popen(start_cmd, **kwargs)
    print(f"Frontend started with PID {proc.pid}")
    return proc


def main():
    root = Path(__file__).resolve().parent
    python_exe = find_venv_python(root)
    print(f"Using Python executable: {python_exe}")

    try:
        backend_proc = run_backend(root, python_exe)
    except Exception as e:
        print(f"Failed to start backend: {e}")
        return 1

    try:
        frontend_proc = run_frontend(root)
    except Exception as e:
        print(f"Failed to start frontend: {e}")
        # Stop backend if frontend fails
        try:
            backend_proc.terminate()
        except Exception:
            pass
        return 1

    # Give frontend a moment to start, then open browser to default port
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        pass

    print("Both processes started. Press Ctrl-C to stop them.")

    try:
        # Wait until both processes exit or user interrupts
        while True:
            if backend_proc.poll() is not None:
                print(f"Backend exited with code {backend_proc.returncode}")
                break
            if frontend_proc.poll() is not None:
                print(f"Frontend exited with code {frontend_proc.returncode}")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received — terminating processes...")
    finally:
        for p, name in ((frontend_proc, 'frontend'), (backend_proc, 'backend')):
            try:
                if p and p.poll() is None:
                    print(f"Terminating {name} (PID {p.pid})...")
                    p.terminate()
                    time.sleep(1)
                    if p.poll() is None:
                        print(f"Killing {name} (PID {p.pid})...")
                        p.kill()
            except Exception as ex:
                print(f"Error stopping {name}: {ex}")

    print("Launcher exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
