"""Compile and run one CUDA file on a Modal GPU, from your Mac.

Run:  modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
      GPU=L4 modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
      modal run kernels/run_cu.py --path kernels/w1/reduce1.cu --ncu   (profile with Nsight Compute)
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
    # -arch=native compiles for whatever GPU this container got
    subprocess.run("nvcc -O3 -arch=native -o /tmp/k /tmp/k.cu", shell=True, check=True)
    # Modal does not allow locking GPU clocks, so --clock-control none.
    # ponytail: profiles one launch of every kernel; add --kernel-name / --set full when you need more
    prefix = "ncu --clock-control none --launch-count 1 --set default --section WarpStateStats " if ncu else ""
    subprocess.run(f"{prefix}/tmp/k {args}", shell=True, check=True)


@app.local_entrypoint()
def main(path: str, args: str = "", ncu: bool = False):
    with open(path) as f:
        compile_and_run.remote(f.read(), args, ncu)
