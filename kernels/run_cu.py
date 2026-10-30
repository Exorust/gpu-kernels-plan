"""Compile and run one CUDA file on a Modal GPU, from your Mac.

Run:  modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
      GPU=L4 modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
      modal run kernels/run_cu.py --path kernels/w1/reduce1.cu --ncu   (profile with Nsight Compute)
--ncu prints the profile and saves kernels/w1/reduce1.ncu-rep on your Mac.
Open that file in the Nsight Compute app: brew install --cask nvidia-nsight-compute
Default GPU is a T4 (cheapest). Use GPU=H100 only for Hopper work (Week 3+).
Compiler errors and your program's printf output appear in your terminal.
"""
import os

import modal

app = modal.App("run-cu")
image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")


@app.function(gpu=os.environ.get("GPU", "T4"), image=image, timeout=600)
def compile_and_run(src: str, args: str = "", ncu: bool = False):
    import subprocess

    with open("/tmp/k.cu", "w") as f:
        f.write(src)
    # -arch=native compiles for whatever GPU this container got; -lineinfo maps SASS to source lines
    subprocess.run("nvcc -O3 -arch=native -lineinfo -o /tmp/k /tmp/k.cu", shell=True, check=True)
    if not ncu:
        subprocess.run(f"/tmp/k {args}", shell=True, check=True)
        return None
    # Modal does not allow locking GPU clocks, so --clock-control none.
    # ponytail: profiles the first launch of each kernel; add --kernel-name to pick one
    subprocess.run(
        f"ncu --clock-control none --launch-count 1 --set full --import-source yes -f -o /tmp/report /tmp/k {args}",
        shell=True, check=True)
    subprocess.run("ncu --import /tmp/report.ncu-rep --page details", shell=True, check=True)
    return open("/tmp/report.ncu-rep", "rb").read()


@app.local_entrypoint()
def main(path: str, args: str = "", ncu: bool = False):
    with open(path) as f:
        report = compile_and_run.remote(f.read(), args, ncu)
    if report:
        out = path.rsplit(".", 1)[0] + ".ncu-rep"
        with open(out, "wb") as f:
            f.write(report)
        print(f"Saved {out}. Open it with: open -a 'NVIDIA Nsight Compute' {out}")
