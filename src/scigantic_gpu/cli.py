"""`scigantic-gpu`: run a Python script on a GPU in Scigantic's cloud.

    scigantic-gpu login --email you@lab.edu      # emails you a code
    scigantic-gpu example                        # writes a starter train.py
    scigantic-gpu run ./scigantic-gpu-example --max-hours 0.25 --follow
    scigantic-gpu run ./my-pipeline --entry train.py --max-hours 1
    scigantic-gpu logs <job> --follow
    scigantic-gpu download <job>

Each run reserves its worst-case cost (instance price x the time limit) against
your account's GPU budget before anything launches, and settles to the minutes
actually billed when it ends. `scigantic-gpu account` shows where you stand,
`scigantic-gpu guide` explains what a script sees.
"""
from __future__ import annotations

import argparse
import fnmatch
import getpass
import os
import sys
import tarfile
import shutil
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from .api import ApiError, Client, api_url, die, load_config, save_config
from .guide import GUIDE

EXCLUDES = [".git", ".hg", "__pycache__", "*.pyc", ".venv", "venv", "node_modules", ".DS_Store", ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"]
MAX_TARBALL_BYTES = 250 * 1024 * 1024
TERMINAL = {"Completed", "Failed", "Stopped"}


def fmt_usd(v) -> str:
    try:
        return f"${float(v):.2f}"
    except (TypeError, ValueError):
        return "-"


def fmt_when(iso) -> str:
    if not iso:
        return "-"
    try:
        dt = datetime.fromisoformat(str(iso).replace("Z", "+00:00")).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return str(iso)


def fmt_duration(start, end) -> str:
    if not start:
        return ""
    try:
        s = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        e = datetime.fromisoformat(str(end).replace("Z", "+00:00")) if end else datetime.now(timezone.utc)
    except ValueError:
        return ""
    secs = int((e - s).total_seconds())
    return f"{secs // 3600}h{(secs % 3600) // 60:02d}m" if secs >= 3600 else f"{secs // 60}m{secs % 60:02d}s"


# ----------------------------------------------------------------------------- login
def cmd_login(a) -> int:
    c = Client(a.api_url, token="")
    try:
        sent = c.request("POST", "/api/gpu/login", {"email": a.email}, auth=False)
    except ApiError as e:
        return die(e.message)
    print(f"Code sent to {sent.get('email', a.email)}. Check your inbox (and spam).")
    code = a.code or input("Enter the code: ").strip()
    if not code:
        return die("no code entered")
    try:
        tok = c.request("POST", "/api/gpu/token", {"email": a.email, "code": code}, auth=False)
    except ApiError as e:
        return die(e.message)
    cfg = load_config()
    cfg.update({"token": tok["token"], "email": tok.get("email", a.email), "expiresAt": tok.get("expiresAt"), "apiUrl": c.base})
    save_config(cfg)
    exp = datetime.fromtimestamp(int(tok["expiresAt"]), tz=timezone.utc).astimezone().strftime("%Y-%m-%d") if tok.get("expiresAt") else "30 days"
    print(f"Logged in as {cfg['email']} (until {exp}). Token saved to {os.environ.get('SCIGANTIC_GPU_CONFIG') or '~/.config/scigantic/gpu.json'}.")
    print()
    print("Next: scigantic-gpu example && scigantic-gpu run ./scigantic-gpu-example --max-hours 0.25 --follow")
    print("      scigantic-gpu guide   (what a script sees, how the cap works)")
    return 0


def cmd_logout(a) -> int:
    cfg = load_config()
    cfg.pop("token", None)
    save_config(cfg)
    print("Logged out.")
    return 0


# ----------------------------------------------------------------------------- example / guide
EXAMPLE_FILES = ("train.py", "requirements.txt")


def cmd_example(a) -> int:
    dest = Path(a.directory)
    src = Path(__file__).parent / "example"
    if dest.exists() and any(dest.iterdir()) and not a.force:
        return die(f"{dest} exists and is not empty; pass --force to overwrite its train.py and requirements.txt")
    dest.mkdir(parents=True, exist_ok=True)
    for name in EXAMPLE_FILES:
        shutil.copyfile(src / name, dest / name)
    print(f"Wrote {dest}/train.py and {dest}/requirements.txt")
    print()
    print("Run it on a GPU (about 5 minutes end to end, a few cents):")
    print(f"  scigantic-gpu run {dest} --max-hours 0.25 --follow")
    print("Or try it locally first (CPU is fine):")
    print(f"  python {dest}/train.py 100")
    return 0


def cmd_guide(a) -> int:
    print(GUIDE, end="")
    return 0


# ----------------------------------------------------------------------------- account
def cmd_account(a) -> int:
    try:
        d = Client(a.api_url).request("GET", "/api/gpu/account")
    except ApiError as e:
        return die(e.message)
    print(f"Account:    {d.get('email', '-')}")
    print(f"GPU budget: {fmt_usd(d['budgetUsd'])}   spent {fmt_usd(d['spentUsd'])}   reserved {fmt_usd(d['reservedUsd'])}   remaining {fmt_usd(d['remainingUsd'])}")
    print(f"Limits:     {d['maxConcurrent']} runs at once, {d['maxRuntimeSeconds'] // 3600} hours per run")
    print()
    print(f"{'instance':<16}{'GPU':<22}{'vCPU':>5}{'RAM':>7}{'$/hour':>9}")
    for i in d.get("instances", []):
        print(f"{i['instanceType']:<16}{i['gpu']:<22}{i['vcpu']:>5}{str(i['memoryGb']) + ' GB':>7}{fmt_usd(i['hourlyUsd']):>9}")
    print()
    print("Reserved = worst case of runs that have not finished; it settles to the billed minutes when they do.")
    return 0


# ----------------------------------------------------------------------------- run
def _excluded(rel: str) -> bool:
    parts = rel.split("/")
    return any(fnmatch.fnmatch(p, pat) for p in parts for pat in EXCLUDES)


def build_tarball(src: Path, out: Path) -> int:
    n = 0
    with tarfile.open(out, "w:gz") as t:
        for root, dirs, files in os.walk(src):
            rel_root = os.path.relpath(root, src)
            dirs[:] = [d for d in dirs if not _excluded(os.path.join(rel_root, d) if rel_root != "." else d)]
            for f in files:
                rel = os.path.normpath(os.path.join(rel_root, f))
                if _excluded(rel):
                    continue
                t.add(os.path.join(root, f), arcname=rel, recursive=False)
                n += 1
    return n


def cmd_run(a) -> int:
    src = Path(a.directory).resolve()
    if not src.is_dir():
        return die(f"{src} is not a directory")
    if not (src / a.entry).is_file():
        return die(f"entry point {a.entry} not found under {src}")
    if a.max_hours <= 0 or a.max_hours > 4:
        return die("--max-hours must be between 0.1 and 4")

    c = Client(a.api_url)
    try:
        acct = c.request("GET", "/api/gpu/account")
    except ApiError as e:
        return die(e.message)
    inst = next((i for i in acct["instances"] if i["instanceType"] == a.instance), None)
    if not inst:
        return die(f"unknown instance {a.instance}; choose one of: " + ", ".join(i["instanceType"] for i in acct["instances"]))
    worst = inst["hourlyUsd"] * a.max_hours * 1.1
    print(f"Run:       python {a.entry} {' '.join(a.args)}".rstrip())
    print(f"Instance:  {a.instance} ({inst['gpu']}), up to {a.max_hours:g} h, worst case about {fmt_usd(worst)}")
    print(f"Budget:    {fmt_usd(acct['remainingUsd'])} of {fmt_usd(acct['budgetUsd'])} remaining")
    if worst > float(acct["remainingUsd"]) + 1e-9:
        return die("this run's worst case exceeds your remaining budget. Shorten --max-hours or wait for running jobs to settle.")
    if not a.yes:
        ans = input("Submit? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Cancelled.")
            return 1

    with tempfile.TemporaryDirectory() as tmp:
        tar_path = Path(tmp) / "code.tar.gz"
        n = build_tarball(src, tar_path)
        size = tar_path.stat().st_size
        if size > MAX_TARBALL_BYTES:
            return die(f"code tarball is {size / 1e6:.0f} MB; keep code under 250 MB and pass data with --input s3://...")
        print(f"Packed {n} files ({size / 1e6:.1f} MB), uploading...")
        try:
            up = c.request("POST", "/api/gpu/upload-url")
            c.put_file(up["uploadUrl"], tar_path)
        except ApiError as e:
            return die(e.message)

    body = {
        "codeS3Uri": up["dataS3Uri"],
        "entryPoint": a.entry,
        "args": list(a.args),
        "instanceType": a.instance,
        "maxRuntimeSeconds": int(a.max_hours * 3600),
        "inputS3Uri": a.input,
        "name": a.name,
    }
    try:
        r = c.request("POST", "/api/gpu/jobs", body)
    except ApiError as e:
        return die(e.message)
    print(f"Submitted {r['jobName']} on {r['instanceType']}; reserved {fmt_usd(r['reservedUsd'])}, {fmt_usd(r['remainingUsd'])} budget left.")
    print(f"Follow it:  scigantic-gpu logs {r['jobName']} --follow")
    print("You will get an email when it finishes.")
    if a.follow:
        a.job = r["jobName"]
        return cmd_logs(a)
    return 0


# ----------------------------------------------------------------------------- jobs / status
def _status_line(j: dict) -> str:
    st = j.get("secondaryStatus") or j.get("status") or "Unknown"
    cost = fmt_usd(j.get("costUsd")) + ("" if j.get("settled") else "*")
    dur = fmt_duration(j.get("startTime"), j.get("endTime"))
    return f"{j['jobName']:<46}{st:<12}{fmt_when(j.get('createdAt')):<18}{dur:<10}{cost:>9}  {j.get('modelName') or ''}"


def cmd_jobs(a) -> int:
    try:
        jobs = Client(a.api_url).request("GET", "/api/gpu/jobs").get("jobs", [])
    except ApiError as e:
        return die(e.message)
    if not jobs:
        print("No GPU runs yet. Start one with: scigantic-gpu run <dir> --entry train.py")
        return 0
    print(f"{'job':<46}{'status':<12}{'submitted':<18}{'ran':<10}{'cost':>9}  name")
    for j in jobs:
        print(_status_line(j))
    print("\n* = reserved worst case; settles to billed minutes when the run ends.")
    return 0


def cmd_status(a) -> int:
    try:
        j = Client(a.api_url).request("GET", f"/api/gpu/jobs/{a.job}")
    except ApiError as e:
        return die(e.message)
    print(f"Job:       {j['jobName']}")
    print(f"Status:    {j.get('status', '-')} / {j.get('secondaryStatus', '-')}")
    print(f"Script:    python {j.get('entryPoint', '-')} {' '.join(j.get('args') or [])}".rstrip())
    print(f"Instance:  {j.get('instanceType', '-')}, limit {int(j.get('maxRuntimeSeconds') or 0) // 60} min")
    print(f"Submitted: {fmt_when(j.get('createdAt'))}   started {fmt_when(j.get('startTime'))}   ended {fmt_when(j.get('endTime'))}")
    if j.get("billableSeconds") is not None:
        print(f"Billed:    {int(j['billableSeconds']) // 60} min")
    print(f"Cost:      {fmt_usd(j.get('costUsd'))} {'(settled)' if j.get('settled') else '(reserved worst case)'}")
    if j.get("exitCode") is not None:
        print(f"Exit code: {j['exitCode']}")
    if j.get("failureReason"):
        print(f"Failure:   {j['failureReason']}")
    if j.get("outputS3Uri"):
        print(f"Output:    scigantic-gpu download {j['jobName']}")
    return 0


# ----------------------------------------------------------------------------- logs
def cmd_logs(a) -> int:
    c = Client(a.api_url)
    token = None
    last_ts = None
    while True:
        path = f"/api/gpu/jobs/{a.job}/logs"
        q = []
        if token:
            q.append(f"nextToken={urllib.request.quote(token)}")
        elif last_ts is not None:
            q.append(f"startTime={last_ts + 1}")
        if q:
            path += "?" + "&".join(q)
        try:
            page = c.request("GET", path)
        except ApiError as e:
            return die(e.message)
        for ev in page.get("events", []):
            print(ev["message"].rstrip("\n"))
            last_ts = ev["timestamp"]
        token = page.get("nextToken")
        if token:
            continue
        if not a.follow:
            return 0
        try:
            j = c.request("GET", f"/api/gpu/jobs/{a.job}")
        except ApiError as e:
            return die(e.message)
        if j.get("status") in TERMINAL:
            print(f"-- {j['status']}; cost {fmt_usd(j.get('costUsd'))}" + (f"; exit code {j['exitCode']}" if j.get("exitCode") is not None else ""))
            return 0 if j.get("status") == "Completed" else 1
        time.sleep(a.interval)


# ----------------------------------------------------------------------------- stop / download
def cmd_stop(a) -> int:
    try:
        r = Client(a.api_url).request("POST", f"/api/gpu/jobs/{a.job}/stop")
    except ApiError as e:
        return die(e.message)
    print(f"{a.job}: {r.get('status', 'Stopping')}. The reservation settles to billed minutes once it is stopped.")
    return 0


def cmd_download(a) -> int:
    try:
        r = Client(a.api_url).request("GET", f"/api/gpu/jobs/{a.job}/download")
    except ApiError as e:
        return die(e.message)
    out = Path(a.out or r.get("filename") or f"{a.job}.tar.gz")
    print(f"Downloading output archive to {out} ...")
    urllib.request.urlretrieve(r["downloadUrl"], out)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB). It holds whatever your script wrote to $SCIGANTIC_OUTPUT_DIR.")
    return 0


# ----------------------------------------------------------------------------- parser
def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="scigantic-gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api-url", help="API base (default: SCIGANTIC_API_URL or https://api.scigantic.com)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("login", help="log in by email; a code is emailed to you")
    s.add_argument("--email", required=True)
    s.add_argument("--code", help="skip the prompt (for scripts)")
    s.set_defaults(fn=cmd_login)

    sub.add_parser("logout", help="forget the saved token").set_defaults(fn=cmd_logout)
    sub.add_parser("guide", help="what a script sees, how the cap works").set_defaults(fn=cmd_guide)

    s = sub.add_parser("example", help="write a starter train.py you can run as-is")
    s.add_argument("directory", nargs="?", default="scigantic-gpu-example", help="where to write it (default: ./scigantic-gpu-example)")
    s.add_argument("--force", action="store_true", help="overwrite an existing directory's files")
    s.set_defaults(fn=cmd_example)
    sub.add_parser("account", help="budget, spend, and the instances you can use").set_defaults(fn=cmd_account)

    s = sub.add_parser(
        "run",
        help="upload a directory and run a script in it on a GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  scigantic-gpu run ./scigantic-gpu-example --max-hours 0.25 --follow
  scigantic-gpu run . --entry pipeline/train.py --max-hours 2 -- --epochs 5 --lr 1e-4
  scigantic-gpu run . --input s3://my-bucket/dataset/ --instance ml.g5.2xlarge --name "fold 3"

inside the script:
  $SCIGANTIC_OUTPUT_DIR   write results here; fetch them with: scigantic-gpu download <job>
  $SCIGANTIC_INPUT_DIR    the --input prefix, downloaded before the script starts
  sys.argv                everything after --

the run stops on its own at --max-hours; its worst case (price x hours x 1.1)
is reserved against your budget first and settles to billed minutes at the end.
""",
    )
    s.add_argument("directory", help="code directory to upload (git, venv and caches are skipped)")
    s.add_argument("--entry", default="train.py", help="script to run, relative to the directory (default: train.py)")
    s.add_argument("--instance", default="ml.g5.xlarge", help="instance type (see: scigantic-gpu account)")
    s.add_argument("--max-hours", type=float, default=1.0, help="hard time limit, up to 4 (default: 1)")
    s.add_argument("--input", help="s3://bucket/prefix/ made available at $SCIGANTIC_INPUT_DIR")
    s.add_argument("--name", help="label shown in job lists and the finish email")
    s.add_argument("-y", "--yes", action="store_true", help="submit without confirming")
    s.add_argument("--follow", action="store_true", help="tail the log after submitting")
    s.add_argument("--interval", type=float, default=15, help=argparse.SUPPRESS)
    s.add_argument("args", nargs="*", help="arguments for the script; put them after -- so they are not read as options")
    s.set_defaults(fn=cmd_run)

    sub.add_parser("jobs", help="list your runs").set_defaults(fn=cmd_jobs)

    s = sub.add_parser("status", help="one run in detail")
    s.add_argument("job")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("logs", help="print what the script wrote to stdout/stderr")
    s.add_argument("job")
    s.add_argument("-f", "--follow", action="store_true", help="keep polling until the run ends")
    s.add_argument("--interval", type=float, default=15, help="seconds between polls with --follow")
    s.set_defaults(fn=cmd_logs)

    s = sub.add_parser("stop", help="stop a running job")
    s.add_argument("job")
    s.set_defaults(fn=cmd_stop)

    s = sub.add_parser("download", help="fetch the output archive of a finished run")
    s.add_argument("job")
    s.add_argument("-o", "--out", help="file to write (default: <job>.tar.gz)")
    s.set_defaults(fn=cmd_download)

    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
