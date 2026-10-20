"""Starter script for `scigantic-gpu run`.

It checks that a GPU is there, trains a tiny network on synthetic data for a
few seconds, and writes two files to the output directory. Everything in that
directory comes back with `scigantic-gpu download <job>`.

Replace the body with your own pipeline. The three things to keep:

  SCIGANTIC_OUTPUT_DIR  write results here (default "." so it also runs locally)
  SCIGANTIC_INPUT_DIR   files from --input s3://bucket/prefix/, if you passed one
  sys.argv              anything after `--` on the run command line
"""
import json
import os
import sys
import time

OUT = os.environ.get("SCIGANTIC_OUTPUT_DIR", ".")
INP = os.environ.get("SCIGANTIC_INPUT_DIR")


def main():
    print("args:", sys.argv[1:])
    if INP and os.path.isdir(INP):
        print("input files:", sorted(os.listdir(INP))[:20])

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("torch", torch.__version__, "| device", device,
          "|", torch.cuda.get_device_name(0) if device == "cuda" else "no GPU")

    # A small regression problem: learn y = sin(x) from noisy samples.
    torch.manual_seed(0)
    x = torch.linspace(-3, 3, 4096, device=device).unsqueeze(1)
    y = torch.sin(x) + 0.1 * torch.randn_like(x)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 64), torch.nn.Tanh(),
        torch.nn.Linear(64, 1),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    steps = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps:
            print(f"step {step}/{steps}  loss {loss.item():.4f}")
    elapsed = time.time() - t0

    os.makedirs(OUT, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    metrics = {"final_loss": round(loss.item(), 5), "steps": steps, "device": device, "seconds": round(elapsed, 2)}
    with open(os.path.join(OUT, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    print("wrote model.pt and metrics.json to", OUT)
    print("metrics:", json.dumps(metrics))


if __name__ == "__main__":
    main()
