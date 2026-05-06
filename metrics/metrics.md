# Adding a New Method to the Metrics Pipeline

When you have a new evaluation run (e.g. `oursaprXX`) and want to compute metrics, you must register it in **two files**. Missing either one causes the pipeline to fail.

---

## Step 1: Create the evaluation directory

Place your generated frames under:

```
diff/evaluations/<method>_<dataset>/
```

Each subfolder is an exposure type (e.g. `auto`, `normal`, `over`, `over20`, `under`, `under5`). Each exposure type contains one subdirectory per video, each containing the predicted HDR frames.

Example layout:
```
diff/evaluations/oursapr28_stuttgart/
    auto/
        bistro_01/   *.exr
        fireplace_01/ *.exr
        ...
    normal/
    over/
    ...
```

---

## Step 2: Add the method name to both scripts

### `compute_metrics_parallel_siddhu.py` — line 34

```python
METHODS = (..., "oursaprXX", ...)
```

This is the low-level compute script. Even though argparse no longer enforces `choices`, keeping `METHODS` up to date documents what's registered.

### `metrics_gathering_sai.py` — line 33

```python
METHODS = (..., "oursaprXX", ...)
```

This is the orchestrator. It reads `METHODS` to discover which method directories to scan and which tasks to schedule. **If you skip this file, the method will never be picked up automatically.**

---

## Step 3: Run metrics

```bash
cd diff/metrics
python metrics_gathering_sai.py --method oursaprXX --dataset stuttgart --gpus 7
```

Or to run all missing methods across all datasets:

```bash
python metrics_gathering_sai.py --all --gpus 0,1,2,3
```

Results land in `diff/metrics/evaluations_output/` as:

```
results_<method>_<dataset>_<type>_17_ds1.csv         # aggregate
results_<method>_<dataset>_<type>_<video>_17_ds1.csv # per-video
```

---

## Ablation methods

Methods whose names start with `ablate` are treated differently during metric computation (different scale normalization, no CRF fitting). No extra flag needed — the prefix is the signal.

---

## Checklist

- [ ] `diff/evaluations/<method>_<dataset>/` exists with the right subfolder structure
- [ ] Method added to `METHODS` in `compute_metrics_parallel_siddhu.py`
- [ ] Method added to `METHODS` in `metrics_gathering_sai.py`
- [ ] Run `metrics_gathering_sai.py --dry-run` to confirm tasks are discovered before launching
