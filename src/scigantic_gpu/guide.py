"""Text for `scigantic-gpu guide`: what a script sees, and how the cap works."""

GUIDE = """\
scigantic-gpu: run a Python script on a GPU in Scigantic's cloud

GETTING STARTED
  scigantic-gpu login --email you@lab.edu      a code is emailed to you
  scigantic-gpu example                        writes ./scigantic-gpu-example
  scigantic-gpu run ./scigantic-gpu-example --max-hours 0.25 --follow
  scigantic-gpu download <job>                 fetch what the script wrote

WHAT RUNS
  The directory you pass to `run` is tarred (skipping .git, virtualenvs and
  caches), uploaded, and unpacked on an NVIDIA A10G. A requirements.txt at the
  top level is pip-installed first. Then your entry point (default train.py)
  runs with CUDA PyTorch 2.6, transformers, pandas, scikit-learn, scipy and
  boto3 already installed.

  Inside the script:
    $SCIGANTIC_OUTPUT_DIR   write results here; they come back via `download`
    $SCIGANTIC_INPUT_DIR    contents of --input s3://bucket/prefix/, if given
    sys.argv                everything after `--` on the run command line

  A non-zero exit code marks the run Failed. `logs` shows stdout and stderr.

COST AND THE CAP
  `account` shows your budget, what you have spent, and what is reserved by
  runs still going. Before a run launches, its worst case (hourly price times
  --max-hours, plus 10 percent) must fit in what is left. When the run ends
  the reservation settles to the minutes actually billed. Runs stop on their
  own at --max-hours; `stop` ends one early. Two runs at once, four hours each.

  You get an email when a run finishes or fails. Ask us to raise your budget
  if the default is too small for real work.

INSTANCES
  ml.g5.xlarge    A10G 24 GB, 4 vCPU, 16 GB RAM    default
  ml.g5.2xlarge   A10G 24 GB, 8 vCPU, 32 GB RAM
  Prices and limits: scigantic-gpu account
"""
