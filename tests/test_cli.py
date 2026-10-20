import os
import tarfile
from pathlib import Path

from scigantic_gpu.cli import build_tarball, fmt_duration, main


def test_tarball_skips_junk(tmp_path: Path):
    src = tmp_path / "code"
    (src / ".git").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text("ref")
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    (src / "pkg").mkdir()
    (src / "pkg" / "model.py").write_text("print('hi')")
    (src / "train.py").write_text("print('hi')")
    out = tmp_path / "code.tar.gz"
    n = build_tarball(src, out)
    with tarfile.open(out) as t:
        names = sorted(t.getnames())
    assert names == ["pkg/model.py", "train.py"]
    assert n == 2


def test_duration_formats():
    assert fmt_duration("2026-01-01T00:00:00Z", "2026-01-01T00:05:30Z") == "5m30s"
    assert fmt_duration("2026-01-01T00:00:00Z", "2026-01-01T01:30:00Z") == "1h30m"
    assert fmt_duration(None, None) == ""


def test_run_rejects_missing_entry(tmp_path: Path, capsys):
    os.environ["SCIGANTIC_GPU_CONFIG"] = str(tmp_path / "cfg.json")
    rc = main(["run", str(tmp_path), "--entry", "nope.py", "-y"])
    assert rc == 1
    assert "entry point nope.py not found" in capsys.readouterr().err


def test_script_args_after_double_dash():
    import argparse
    from scigantic_gpu import cli
    captured = {}
    cli.cmd_run = lambda a: captured.update(vars(a)) or 0
    rc = main(["run", ".", "--entry", "train.py", "-y", "--", "--epochs", "3"])
    assert rc == 0
    assert captured["entry"] == "train.py" and captured["yes"] is True
    assert captured["args"] == ["--epochs", "3"]


def test_example_writes_runnable_starter(tmp_path: Path, capsys):
    dest = tmp_path / "ex"
    assert main(["example", str(dest)]) == 0
    assert (dest / "train.py").is_file() and (dest / "requirements.txt").is_file()
    assert main(["example", str(dest)]) == 1  # refuses to clobber
    assert "--force" in capsys.readouterr().err
    assert main(["example", str(dest), "--force"]) == 0


def test_guide_prints(capsys):
    assert main(["guide"]) == 0
    out = capsys.readouterr().out
    assert "SCIGANTIC_OUTPUT_DIR" in out and "COST AND THE CAP" in out
