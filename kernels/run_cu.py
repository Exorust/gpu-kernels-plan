"""Compile and run one CUDA file on a Modal H100, from your Mac.

Run:  modal run kernels/run_cu.py --path kernels/w1/reduce1.cu
Compiler errors and your program's printf output appear in your terminal.
"""
import modal

app = modal.App("run-cu")
image = modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")


@app.function(gpu="H100", image=image, timeout=600)
def compile_and_run(src: str, args: str = ""):
    import subprocess

    with open("/tmp/k.cu", "w") as f:
        f.write(src)
    subprocess.run("nvcc -O3 -arch=sm_90 -o /tmp/k /tmp/k.cu", shell=True, check=True)
    subprocess.run(f"/tmp/k {args}", shell=True, check=True)


@app.local_entrypoint()
def main(path: str, args: str = ""):
    with open(path) as f:
        compile_and_run.remote(f.read(), args)
