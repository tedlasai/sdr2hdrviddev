#!/usr/bin/env python3
import argparse
import math
import time
import sys

def human_gb(x_bytes: int) -> float:
    return x_bytes / (1024**3)

def pick_square_n(target_bytes: int, dtype_bytes: int, num_mats: int = 3) -> int:
    # We allocate A, B, and C for C = A @ B  => ~3 * n*n*dtype_bytes
    # target_bytes ≈ num_mats * n^2 * dtype_bytes
    n = int(math.sqrt(target_bytes / (num_mats * dtype_bytes)))
    return max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-gb", type=float, default=40.0,
                    help="Target VRAM to occupy (GB). Will clamp to available free VRAM.")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16",
                    help="Compute dtype. fp16 is fastest on most GPUs.")
    ap.add_argument("--margin-gb", type=float, default=2.0,
                    help="Leave this much free VRAM (GB) to reduce OOM risk.")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="Run for N seconds (0 means run forever).")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Warmup iterations before reporting.")
    ap.add_argument("--report-every", type=int, default=50,
                    help="Report every N iterations.")
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print("Failed to import torch. Install PyTorch with CUDA support.", file=sys.stderr)
        raise

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Make sure PyTorch sees your GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    dtype_bytes = torch.tensor([], dtype=dtype).element_size()

    free_b, total_b = torch.cuda.mem_get_info()
    free_gb = human_gb(free_b)
    total_gb = human_gb(total_b)

    # Clamp target to (free - margin)
    target_gb = min(args.target_gb, max(free_gb - args.margin_gb, 0.5))
    target_bytes = int(target_gb * (1024**3))

    print(f"GPU total: {total_gb:.2f} GB | free now: {free_gb:.2f} GB")
    print(f"Target allocation: {target_gb:.2f} GB (dtype={args.dtype}, margin={args.margin_gb:.2f} GB)")

    # Try allocating A, B, C as square matrices.
    # If OOM happens, we reduce n and retry.
    n = pick_square_n(target_bytes, dtype_bytes, num_mats=3)

    # Make sizes multiples of 256 for better tensor core alignment when possible.
    def align(x, a=256):
        return max((x // a) * a, a)

    n = align(n, 256)
    print(f"Initial matrix size: n={n} (each matrix ~ {human_gb(n*n*dtype_bytes):.2f} GB)")

    A = B = C = None
    while True:
        try:
            torch.cuda.empty_cache()
            A = torch.empty((n, n), device=device, dtype=dtype)
            B = torch.empty((n, n), device=device, dtype=dtype)
            C = torch.empty((n, n), device=device, dtype=dtype)
            # Touch memory so allocation becomes real
            A.normal_()
            B.normal_()
            C.zero_()
            torch.cuda.synchronize()
            break
        except torch.cuda.OutOfMemoryError:
            n = int(n * 0.9)
            n = align(n, 256)
            if n <= 256:
                raise RuntimeError("OOM even at very small n. Reduce --target-gb or increase --margin-gb.")
            print(f"OOM. Retrying with n={n} ...")

    alloc_est = 3 * n * n * dtype_bytes
    print(f"Allocated approx: {human_gb(alloc_est):.2f} GB across A/B/C")

    # Keep GPU busy with repeated GEMM
    torch.backends.cuda.matmul.allow_tf32 = True  # helps fp32 throughput on Ampere+
    start = time.time()
    it = 0

    # Warmup
    for _ in range(args.warmup):
        C = A @ B
    torch.cuda.synchronize()

    print("Running... (Ctrl+C to stop)")
    last_t = time.time()

    try:
        while True:
            # One heavy op per loop
            C = A @ B

            it += 1
            if it % args.report_every == 0:
                now = time.time()
                dt = now - last_t
                elapsed = now - start
                # Rough FLOP count for GEMM: 2*n^3
                tflops = (2.0 * (n**3) / dt) / 1e12
                free_b2, total_b2 = torch.cuda.mem_get_info()
                print(
                    f"iter={it:8d} | ~{tflops:6.2f} TFLOP/s (rough) | "
                    f"free={human_gb(free_b2):5.2f} GB | elapsed={elapsed:6.1f}s"
                )
                last_t = now


    except KeyboardInterrupt:
        print("\nStopped by user.")

if __name__ == "__main__":
    main()
