#!/usr/bin/env python3
"""bench_runner.py

Simple benchmarking utility to run the same command multiple times with
variable-length random input and record timings.

Clean, single-version bench runner.

python .\app-back\bench_runner.py --mode api --input-mode text --api-base "http://127.0.0.1:8000" --repetitions 1 --min-length 50 --max-length 50 --metrics --timeout 300
"""
from __future__ import annotations

import argparse
import os
import csv
import datetime
import json
import random
import string
import subprocess
import tempfile
import time
from pathlib import Path
import statistics
import sys
import uuid
try:
    from metrics_logger import log_metrics as _ml_log_metrics, CSV_COLUMNS as _ML_COLUMNS
except Exception:
    try:
        from .metrics_logger import log_metrics as _ml_log_metrics, CSV_COLUMNS as _ML_COLUMNS
    except Exception:
        _ml_log_metrics = None
        _ML_COLUMNS = None


LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
BENCH_CSV = LOGS_DIR / "benchmarks.csv"


def random_text(length: int) -> str:
    letters = string.ascii_letters + "     "
    return "".join(random.choice(letters) for _ in range(length))


def run_single(cmd_template: str, input_text: str | None, timeout: int | None = None, record_proc_metrics: bool = False, input_path: str | None = None) -> dict:
    # If an explicit input_path is provided (e.g. audio/video file), use it
    created_temp = False
    if input_path is None:
        # create a temporary text input file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8", suffix=".txt") as fh:
            fh.write(input_text or "")
            input_path = fh.name
            created_temp = True

    cmd = cmd_template.replace("{input_file}", input_path)
    start = datetime.datetime.utcnow()
    proc_metrics: dict | None = None
    stdout = ""
    stderr = ""
    rc = -1

    if record_proc_metrics:
        try:
            import psutil
        except Exception:
            psutil = None
        if psutil is not None:
            # Use Popen so we can sample process metrics while it runs
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            peak_rss = None
            cpu_samples: list[float] = []
            try:
                proc_ps = psutil.Process(p.pid)
                # prime cpu_percent
                try:
                    proc_ps.cpu_percent(None)
                except Exception:
                    pass
            except Exception:
                proc_ps = None

            start_poll = time.time()
            try:
                while True:
                    if p.poll() is not None:
                        break
                    if timeout is not None and (time.time() - start_poll) > timeout:
                        try:
                            p.kill()
                        except Exception:
                            pass
                        rc = -1
                        stdout = ""
                        stderr = "TIMEOUT"
                        break
                    if proc_ps is not None:
                        try:
                            cpu = proc_ps.cpu_percent(None)
                            mem = proc_ps.memory_info().rss
                            cpu_samples.append(cpu)
                            if peak_rss is None or mem > peak_rss:
                                peak_rss = mem
                        except psutil.NoSuchProcess:
                            break
                        except Exception:
                            pass
                    time.sleep(0.05)
                # get remaining output
                try:
                    out, err = p.communicate(timeout=1)
                    stdout = out[:1000] if out else ""
                    stderr = err[:1000] if err else ""
                    if rc != -1:
                        rc = p.returncode
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
                    stdout = ""
                    stderr = "ERROR: collecting output"
                    rc = -1

            finally:
                avg_cpu = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None
                proc_metrics = {"peak_memory_bytes": peak_rss, "avg_cpu_percent": avg_cpu, "cpu_samples": len(cpu_samples)}
        else:
            # psutil not available: fall back to simple subprocess.run
            try:
                proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
                rc = proc.returncode
                stdout = proc.stdout[:1000] if proc.stdout else ""
                stderr = proc.stderr[:1000] if proc.stderr else ""
            except subprocess.TimeoutExpired as e:
                rc = -1
                stdout = ""
                stderr = f"TIMEOUT: {e}"
            except Exception as e:
                rc = -1
                stdout = ""
                stderr = f"ERROR: {e}"
    else:
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            rc = proc.returncode
            stdout = proc.stdout[:1000] if proc.stdout else ""
            stderr = proc.stderr[:1000] if proc.stderr else ""
        except subprocess.TimeoutExpired as e:
            rc = -1
            stdout = ""
            stderr = f"TIMEOUT: {e}"
        except Exception as e:
            rc = -1
            stdout = ""
            stderr = f"ERROR: {e}"

    end = datetime.datetime.utcnow()
    duration = (end - start).total_seconds()

    return {
        "timestamp_utc": start.isoformat(),
        "exit_code": rc,
        "duration_seconds": duration,
        "stdout_snippet": stdout.replace("\n", " ") if stdout else "",
        "stderr_snippet": stderr.replace("\n", " ") if stderr else "",
        "input_path": input_path,
        "created_temp": created_temp,
        "proc_metrics": json.dumps(proc_metrics, ensure_ascii=False) if proc_metrics is not None else "",
    }


def append_csv_row(path: Path, header: list[str], row: list):
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", required=False, help="Command template to run. Use {input_file} placeholder.")
    p.add_argument("--repetitions", type=int, default=10, help="Number of repetitions")
    p.add_argument("--min-length", type=int, default=50, help="Minimum input length (chars)")
    p.add_argument("--max-length", type=int, default=100, help="Maximum input length (chars)")
    p.add_argument("--timeout", type=int, default=300, help="Timeout per run in seconds")
    p.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    p.add_argument("--out", type=str, default=str(BENCH_CSV), help="Output CSV path")
    p.add_argument("--metrics", action="store_true", help="If set, read new rows from runtime_metrics.csv after each run and attach them")
    p.add_argument("--proc-metrics", action="store_true", help="Measure process CPU/memory for local runs (requires psutil).")
    p.add_argument("--keep-inputs", action="store_true", help="If set, keep generated input files under logs/inputs/ for debugging")
    p.add_argument("--mode", choices=["local", "api"], default="local", help="Run local command template (default) or call backend API endpoints")
    p.add_argument("--youtube-url", type=str, default="https://www.youtube.com/watch?v=dQw4w9WgXcQ", help="YouTube URL to use when input-mode is youtube")
    p.add_argument("--api-base", type=str, default="http://127.0.0.1:8000", help="Base URL of the backend API when --mode api is used")
    p.add_argument("--input-mode", choices=["text", "youtube", "audio", "video"], default="text", help="When --mode api: use text input (generate random text), youtube url, audio or video file")
    args = p.parse_args(argv)

    if args.mode == "local" and not args.cmd:
        print("--cmd is required when --mode is local", file=sys.stderr)
        sys.exit(2)

    if args.seed is not None:
        random.seed(args.seed)

    out_path = Path(args.out)
    header = [
        "timestamp_utc",
        "run_index",
        "input_length_chars",
        "cmd_template_or_mode",
        "exit_code",
        "duration_seconds",
        "stdout_snippet",
        "stderr_snippet",
        "pipeline_metrics_json",
        "proc_metrics_json",
        "input_path",
    ]

    durations: list[float] = []

    for i in range(1, args.repetitions + 1):
        length = random.randint(args.min_length, args.max_length)
        txt = random_text(length)
        print(f"Starting run {i}/{args.repetitions}...")

        # If metrics collection is requested, note current number of lines in runtime_metrics
        pre_count = 0
        runtime_csv = Path(__file__).resolve().parent / "logs" / "runtime_metrics.csv"
        if args.metrics and runtime_csv.exists():
            try:
                with runtime_csv.open("r", encoding="utf-8") as fh:
                    pre_count = len([ln for ln in fh.read().splitlines() if ln.strip()])
            except Exception:
                pre_count = 0

        if args.mode == "local":
            # run local command; optionally record process metrics
            print(f"  Mode=local. Running command template: {args.cmd}")
            if args.input_mode in ("audio", "video"):
                base = Path(__file__).resolve().parent
                fname = base / "utils" / ("videoplayback.m4a" if args.input_mode == "audio" else "videoplayback.mp4")
                if not fname.exists():
                    print(f"  ERROR: input file not found: {fname}", file=sys.stderr)
                    result = {
                        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
                        "exit_code": -1,
                        "duration_seconds": 0.0,
                        "stdout_snippet": "",
                        "stderr_snippet": f"input file not found: {fname}",
                        "input_path": str(fname),
                        "created_temp": False,
                    }
                else:
                    result = run_single(args.cmd, None, timeout=args.timeout, record_proc_metrics=args.proc_metrics, input_path=str(fname))
                    print(f"  Command finished: rc={result.get('exit_code')} duration={result.get('duration_seconds'):.3f}s")
            else:
                result = run_single(args.cmd, txt, timeout=args.timeout, record_proc_metrics=args.proc_metrics)
                print(f"  Command finished: rc={result.get('exit_code')} duration={result.get('duration_seconds'):.3f}s")
        else:
            # API mode: call backend endpoints instead of running a local command
            def run_api_call(api_base: str, input_mode: str, text_val: str, youtube_url: str, timeout_sec: int, request_id: str) -> dict:
                start = datetime.datetime.utcnow()
                rc = -1
                stdout = ""
                stderr = ""
                input_path = ""
                try:
                    import requests
                except Exception:
                    requests = None

                try:
                    if input_mode == "text":
                        url = api_base.rstrip("/") + "/generate_from_text/"
                        payload = {"text": text_val, "request_id": request_id}
                        if requests is not None:
                            r = requests.post(url, json=payload, timeout=timeout_sec)
                            rc = 0 if r.status_code < 400 else r.status_code
                            stdout = r.text
                        else:
                            import urllib.request, urllib.error
                            data = json.dumps(payload).encode("utf-8")
                            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                                stdout = resp.read().decode("utf-8")
                                rc = 0
                    elif input_mode == "youtube":
                        url = api_base.rstrip("/") + "/transcribe_youtube/"
                        payload = {"url": youtube_url, "request_id": request_id}
                        if requests is not None:
                            r = requests.post(url, json=payload, timeout=timeout_sec)
                            rc = 0 if r.status_code < 400 else r.status_code
                            stdout = r.text
                        else:
                            import urllib.request, urllib.error
                            data = json.dumps(payload).encode("utf-8")
                            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                            with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
                                stdout = resp.read().decode("utf-8")
                                rc = 0
                    elif input_mode in ("audio", "video"):
                        base = Path(__file__).resolve().parent
                        fname = base / "utils" / ("videoplayback.m4a" if input_mode == "audio" else "videoplayback.mp4")
                        input_path = str(fname)
                        if not fname.exists():
                            rc = -1
                            stderr = f"file not found: {fname}"
                        else:
                            url = api_base.rstrip("/") + "/process_video/"
                            if requests is None:
                                rc = -1
                                stderr = "requests library required for file upload"
                            else:
                                with open(fname, "rb") as fh:
                                    files = {"file": (fname.name, fh, "audio/m4a" if input_mode == "audio" else "video/mp4")}
                                    data = {"request_id": request_id}
                                    r = requests.post(url, files=files, data=data, timeout=timeout_sec)
                                    # if Spanish-localized endpoint exists, try fallback
                                    if r.status_code >= 400:
                                        try:
                                            url2 = api_base.rstrip("/") + "/procesar_video/"
                                            r2 = requests.post(url2, files=files, data=data, timeout=timeout_sec)
                                            r = r2
                                        except Exception:
                                            pass
                                    rc = 0 if r.status_code < 400 else r.status_code
                                    stdout = r.text
                except Exception as e:
                    rc = -1
                    stdout = ""
                    stderr = str(e)

                end = datetime.datetime.utcnow()
                duration = (end - start).total_seconds()
                return {
                    "timestamp_utc": start.isoformat(),
                    "exit_code": rc,
                    "duration_seconds": duration,
                    "stdout_snippet": stdout.replace("\n", " ") if stdout else "",
                    "stderr_snippet": stderr.replace("\n", " ") if stderr else "",
                    "input_path": input_path,
                    "request_id": request_id,
                }

            api_request_id = str(uuid.uuid4())
            print(f"  Mode=api. Calling {args.api_base} endpoint, request_id={api_request_id}")
            result = run_api_call(args.api_base, args.input_mode, txt, args.youtube_url, args.timeout, api_request_id)
            print(f"  API call returned: rc={result.get('exit_code')} duration={result.get('duration_seconds'):.3f}s request_id={result.get('request_id')}")

        # Append duration to stats
        durations.append(result["duration_seconds"])

        # Optionally move input to logs/inputs/ for inspection
        kept_input = ""
        if args.keep_inputs:
            inputs_dir = LOGS_DIR / "inputs"
            inputs_dir.mkdir(parents=True, exist_ok=True)
            target = inputs_dir / f"input_run_{int(time.time())}_{i}.txt"
            try:
                # If the run produced or referenced an input file, try to copy it
                ip = result.get("input_path")
                if ip:
                    try:
                        import shutil
                        shutil.copy(ip, target)
                        kept_input = str(target)
                    except Exception:
                        # fallback: if local run, try atomic replace; else write text
                        try:
                            if args.mode == "local":
                                Path(ip).replace(target)
                                kept_input = str(target)
                            else:
                                target.write_text(txt, encoding="utf-8")
                                kept_input = str(target)
                        except Exception:
                            kept_input = ip
                else:
                    # no input file recorded; write the text used for the run
                    target.write_text(txt or "", encoding="utf-8")
                    kept_input = str(target)
            except Exception:
                kept_input = result.get("input_path", "")
        else:
            # cleanup only temporary inputs we created locally
            try:
                if args.mode == "local" and result.get("created_temp"):
                    Path(result["input_path"]).unlink()
            except Exception:
                pass

        # Build or obtain pipeline metrics matching `metrics_logger.CSV_COLUMNS`.
        pipeline_metrics = ""
        if args.metrics:
            # If running local commands, synthesize a metrics record consistent
            # with `metrics_logger.CSV_COLUMNS` and append it using log_metrics()
            if args.mode == "local":
                try:
                    # local import from same package
                    from metrics_logger import log_metrics, CSV_COLUMNS
                except Exception:
                    try:
                        from .metrics_logger import log_metrics, CSV_COLUMNS
                    except Exception:
                        log_metrics = None
                        CSV_COLUMNS = None

                rec: dict = {}
                rec["timestamp_utc"] = datetime.datetime.utcnow().isoformat()
                rec["request_id"] = str(uuid.uuid4())
                rec["endpoint"] = args.cmd if args.mode == "local" else args.api_base
                rec["input_type"] = "text" if args.mode == "local" else args.input_mode
                rec["input_length_seconds"] = ""
                rec["text_length_chars"] = length
                rec["n_letters_rendered"] = sum(1 for ch in txt if ch.isalpha())
                rec["success"] = True if result.get("exit_code", -1) == 0 else False
                rec["error_stage"] = "" if rec["success"] else "run"
                rec["error_message_short"] = result.get("stderr_snippet", "")[:1000]
                rec["output_video_filename"] = ""
                rec["output_video_duration_seconds"] = ""
                rec["n_poses_rendered"] = ""
                rec["t_total"] = result.get("duration_seconds")
                rec["t_download"] = ""
                rec["t_extract_audio"] = ""
                rec["t_asr"] = ""
                rec["t_text_normalisation"] = ""
                rec["t_pose_sequence"] = ""
                rec["t_render"] = ""

                # Try to write into runtime_metrics.csv so API-mode readers see it too
                if _ml_log_metrics is not None:
                    try:
                        _ml_log_metrics(rec)
                    except Exception:
                        pass

                try:
                    pipeline_metrics = json.dumps(rec, ensure_ascii=False)
                except Exception:
                    pipeline_metrics = ""
            else:
                # API mode: poll runtime_metrics.csv for new rows appended by the backend
                # and try to match by request_id returned by the API call.
                runtime_csv = Path(__file__).resolve().parent / "logs" / "runtime_metrics.csv"
                matched_row = None
                start_wait = time.time()
                last_read = pre_count
                req_id = result.get("request_id")
                print(f"  Polling for runtime metrics (request_id={req_id}) for up to {args.timeout}s...")
                while time.time() - start_wait < args.timeout:
                    try:
                        if runtime_csv.exists():
                            with runtime_csv.open("r", encoding="utf-8") as fh:
                                lines = [ln for ln in fh.read().splitlines() if ln.strip()]
                            if len(lines) > last_read:
                                try:
                                    rdr = csv.DictReader(lines)
                                    rows = list(rdr)
                                except Exception:
                                    rows = []
                                # look for a row with matching request_id
                                if req_id:
                                    for r in rows:
                                        if r.get("request_id") == req_id:
                                            matched_row = r
                                            break
                                # fallback: if no request_id match, take last appended row(s)
                                if matched_row is None and len(rows) >= last_read:
                                    try:
                                        matched_row = rows[-1]
                                    except Exception:
                                        matched_row = None
                                if matched_row is not None:
                                    break
                    except Exception:
                        pass
                    time.sleep(0.5)
                if matched_row:
                    try:
                        pipeline_metrics = json.dumps(matched_row, ensure_ascii=False)
                    except Exception:
                        pipeline_metrics = ""

        row = [
            result["timestamp_utc"],
            i,
            length,
            args.cmd if args.mode == "local" else f"api:{args.input_mode}",
            result["exit_code"],
            f"{result['duration_seconds']:.6f}",
            result["stdout_snippet"],
            result["stderr_snippet"],
            pipeline_metrics,
            result.get("proc_metrics", ""),
            kept_input or result.get("input_path", ""),
        ]
        append_csv_row(out_path, header, row)
        print(f"Run {i}/{args.repetitions}: length={length} duration={result['duration_seconds']:.3f}s rc={result['exit_code']}")

    # Summary
    mean_d = statistics.mean(durations) if durations else 0.0
    median_d = statistics.median(durations) if durations else 0.0
    stdev_d = statistics.stdev(durations) if len(durations) > 1 else 0.0
    min_d = min(durations) if durations else 0.0
    max_d = max(durations) if durations else 0.0

    summary = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "repetitions": args.repetitions,
        "mean_duration": mean_d,
        "median_duration": median_d,
        "stdev_duration": stdev_d,
        "min_duration": min_d,
        "max_duration": max_d,
        "cmd_template": args.cmd if args.mode == "local" else f"api:{args.input_mode}",
        "min_length": args.min_length,
        "max_length": args.max_length,
    }

    # If metrics were collected, compute averages for numeric fields found in
    # the pipeline_metrics_json across runs and add them to the summary.
    if args.metrics:
        numeric_fields = [
            "input_length_seconds",
            "text_length_chars",
            "n_letters_rendered",
            "output_video_duration_seconds",
            "n_poses_rendered",
            "t_total",
            "t_download",
            "t_extract_audio",
            "t_asr",
            "t_text_normalisation",
            "t_pose_sequence",
            "t_render",
        ]
        accum: dict[str, list[float]] = {k: [] for k in numeric_fields}
        try:
            with out_path.open("r", encoding="utf-8") as fh:
                rdr = csv.DictReader(fh)
                for r in rdr:
                    pm = r.get("pipeline_metrics_json")
                    if not pm:
                        continue
                    try:
                        d = json.loads(pm)
                    except Exception:
                        continue
                    for k in numeric_fields:
                        v = d.get(k)
                        if v in (None, ""):
                            continue
                        try:
                            fv = float(v)
                            accum[k].append(fv)
                        except Exception:
                            pass
            for k, vals in accum.items():
                summary[f"mean_{k}"] = statistics.mean(vals) if vals else None
        except Exception:
            pass

    summary_path = out_path.with_name(out_path.stem + "_summary.csv")
    s_header = list(summary.keys())
    s_row = [summary[k] for k in s_header]
    append_csv_row(summary_path, s_header, s_row)

    print("\nSummary:")
    print(f"  runs: {args.repetitions}")
    print(f"  mean: {mean_d:.6f}s  median: {median_d:.6f}s  stdev: {stdev_d:.6f}s")
    print(f"  min: {min_d:.6f}s  max: {max_d:.6f}s")
    print(f"Results appended to: {out_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
