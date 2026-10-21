# scigantic-gpu

Run your own Python script on a GPU in Scigantic's cloud, from the terminal.
No AWS account, no credentials, no cluster to manage. Each run is capped by a
time limit you set and by a dollar budget on your Scigantic account.

```
uv tool install scigantic-gpu            # or: pipx install scigantic-gpu
scigantic-gpu login --email you@lab.edu  # a code lands in your inbox
scigantic-gpu example                    # writes ./scigantic-gpu-example/train.py
scigantic-gpu run ./scigantic-gpu-example --max-hours 0.25 --follow
scigantic-gpu download ft-gpu-...
```

`uv` and `pipx` put the command on your PATH in its own environment, which
is what Homebrew and Debian Pythons require. Plain `pip install scigantic-gpu`
works inside a virtualenv. The tool has no dependencies.

You need a Scigantic account for the email you log in with, and GPU access
turned on for it. Sign up at https://scigantic.com/signup, then write to
aaron@scigantic.com to request access. `scigantic-gpu account` shows your budget.

The example trains a tiny network for a few seconds and writes `model.pt`
and `metrics.json` to the output directory. Its `train.py` is commented and
meant to be replaced with your own code; `scigantic-gpu guide` prints the
short version of everything below.

## What runs

The directory you pass to `run` is tarred (skipping `.git`, virtualenvs and
caches), uploaded, and unpacked on an NVIDIA A10G instance. If it has a
`requirements.txt` at the top level, it is pip-installed first. Then your
entry point runs with the same Python that Scigantic's fine-tuning jobs use:
CUDA PyTorch 2.6, transformers, pandas, scikit-learn, scipy, boto3.

Inside the script:

- `$SCIGANTIC_OUTPUT_DIR` is where to write results. Everything there comes
  back as a tarball through `scigantic-gpu download`.
- `$SCIGANTIC_INPUT_DIR` holds the contents of `--input s3://bucket/prefix/`
  if you passed one. Public buckets work; Scigantic's mirrored archives are
  public.
- Arguments after `--` reach `sys.argv`:
  `scigantic-gpu run . --entry train.py -- --epochs 3 --lr 1e-4`

A non-zero exit code marks the run Failed and shows up in `status`.

## Cost and the cap

`scigantic-gpu account` prints your budget, what you have spent, and what is
reserved by runs still going. Before a run launches, its worst case (hourly
price times `--max-hours`, plus a 10 percent margin) must fit in what is
left. When the run ends the reservation settles to the minutes AWS billed.
Runs stop on their own at `--max-hours`; `scigantic-gpu stop` ends one early.

You get an email when a run finishes or fails. Budgets are set per account
and are a lifetime total; write to us to raise yours.

## Your own script

Any directory with a Python entry point works. Keep it under 250 MB and pass
data with `--input`. A `requirements.txt` at the top level is installed
before the script runs; torch, transformers, pandas, scikit-learn, scipy and
boto3 are already there. Write results to `$SCIGANTIC_OUTPUT_DIR` (fall back
to `.` so the same script runs on your laptop). Exit non-zero on failure so
the run is marked Failed.

## Commands

| command | what it does |
| --- | --- |
| `login --email` | email a code, exchange it for a 30-day token saved in `~/.config/scigantic/gpu.json` |
| `example [DIR]` | write a starter `train.py` and `requirements.txt` you can run as-is |
| `guide` | what a script sees and how the cap works, in the terminal |
| `account` | budget, spend, allowed instances and prices |
| `run DIR` | upload and launch; `--entry`, `--instance`, `--max-hours`, `--input`, `--name`, `-y`, `--follow` |
| `jobs` | your runs with status and cost |
| `status JOB` | one run in detail |
| `logs JOB [-f]` | the script's stdout and stderr |
| `stop JOB` | stop a run |
| `download JOB` | fetch the output archive |
| `logout` | forget the saved token |

Set `SCIGANTIC_API_URL` to point at a different API (staging).

## Questions

Open an issue here or write to aaron@scigantic.com.
