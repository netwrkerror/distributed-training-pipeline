# distributed-training-pipeline

A distributed training pipeline for nuScenes camera crops, built from the primitives up.

Milestones A and B are **pure PyTorch DDP with no Ray** — deliberately. The goal is to
understand what higher-level frameworks abstract before adopting them, so `torch.distributed`,
`DistributedSampler`, checkpointing and the input pipeline are all built and measured directly.
Ray Train, and possibly FSDP, land later as additional backends over the same data and training
layer.

This is a learning repository with real measurements, not a demo. Where a result is inside the
noise floor, it is reported as such.

**Sibling repo:** [`ray-multimodal-pipeline`](https://github.com/netwrkerror/ray-multimodal-pipeline)
— the same nuScenes data, but *inference* (Ray Data + Faster R-CNN batch inference). That one is
inference, this one is training. `src/dtp/manifest.py` is copied from it, with a note saying so.

## Status

| Milestone | | |
|---|---|---|
| **A — DDP mechanics on CPU** | | |
| A1 | Repo scaffold and dev environment | ✅ |
| A2 | Dataset layer: nuScenes crops | ✅ |
| A3 | Single-process training baseline | ✅ |
| A4 | DDP wrapper | ✅ |
| A5 | DistributedSampler and the no-duplicate assertion | ✅ |
| A6 | Checkpointing | ✅ |
| A7 | Resume | ⬜ |
| A8 | Fault injection and recovery | ⬜ |
| **B — Measure and fix the input pipeline** | | |
| B1 | DataLoader-only throughput harness | ⬜ |
| B2 | Find and explain the knee | ⬜ |
| B3 | `persistent_workers` on short epochs | ⬜ |
| B4 | Shard repacker | ⬜ |
| B5 | Iterable dataset over shards | ⬜ |
| B6 | Before/after benchmark and write-up | ⬜ |

Deferred on purpose: CUDA, NCCL, multi-node, FSDP, mixed precision, Ray Train.

## Quickstart

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). CPU only.

```bash
make install    # create the venv and install dependencies
make doctor     # check the host for defects that make distributed runs hang
make hello      # 4-process gloo hello-world: rank, world_size, and a checked all_reduce
make train      # single-process training baseline
make ddp        # the same loop across 4 gloo processes
make check-ddp  # assert DDP really averages gradients across ranks
make check-sampler  # assert the ranks partition the dataset exactly
make test       # fast tests
make test-all   # everything, including the 4-process run
```

To build the dataset you need the nuScenes **v1.0-mini** split on disk (~4GB, not included):

```bash
make crops DTP_NUSCENES_ROOT=/path/to/nuscenes
```

That writes `data/crops.jsonl` — one record per object crop, naming an image file and a pixel
box inside it — plus a `.classes.json` sidecar. 4,104 crops across 10 classes from 404 CAM_FRONT
keyframes, with the long-tailed distribution you would expect (car 48.0%, trailer 0.4%).

## Notes from the work so far

**A corrupt checkpoint does not announce itself.** `torch.save` writes a zip container, so a
*truncated* checkpoint is caught by the format. A single flipped byte is not: the file loads
without complaint, the weights come back finite and plausibly scaled, and training resumes from
silently wrong parameters. Measured on a real checkpoint — byte flips at 50% and 90% of the file
both loaded happily. Checkpoints here record a sha256 in the marker and verify it on load.

**Checkpoint writes are atomic**: temp file in the destination directory → `fsync` the file →
`os.replace` → `fsync` the directory. The naive `torch.save(state, "latest.pt")` overwrites the
only good copy in place, so a crash during the write destroys the checkpoint you were keeping in
order to survive crashes.

**Sharding is where distribution finally pays.** Without `DistributedSampler` all four ranks
iterate the whole dataset: 32.6s per epoch to reproduce the single-process result exactly. With
it, each rank takes a disjoint 802-record shard and the epoch drops to 7.2s — 2.8× faster than
the 20.4s single-process baseline.

**`DistributedSampler` pads, so "no duplicates" is not quite true.** Every rank must run the same
number of steps, or one reaches the next collective alone and blocks. When the dataset does not
divide evenly, `drop_last=False` repeats records from the front of the permutation: coverage is
complete but `world_size - (n % world_size)` records appear twice in one epoch. The real split
hides this (3208 and 896 are both divisible by 4); the tests assert the general case.

**Learning-rate scaling made convergence worse, and it is reported that way.** At a real
effective batch of 256, six epochs each: unscaled `3e-3` → 0.6031, √-scaled `6e-3` → 0.6679,
linearly scaled `1.2e-2` → 0.7174. Monotonically worse with more scaling. The linear rule is SGD
folklore that assumes a warmup this run does not have, and AdamW already decouples step size from
gradient magnitude. Left unscaled by default.

**Most of DDP's apparent slowdown was a launcher default, not the framework.** A naive
before/after says DDP cost 2.4× per step (120.3ms → 284.6ms). Decomposed: single-process with
one thread is already 226.9ms, because `torchrun` sets `OMP_NUM_THREADS=1` while a bare
`python` run used seven threads. So 1.9× is threading and only 1.25× is gradient
synchronisation. Comparing a launched run against an unlaunched one measures the launcher.

**"Per-rank losses agree" does not prove gradients are synchronised.** With every rank seeing
identical data from identical initial weights, the losses agree whether or not the all-reduce
runs at all — so the obvious done-when is satisfiable by a broken implementation. `make check-ddp`
asserts the real property: ranks given deliberately *different* data end up with bit-identical
gradients equal to the mean of their independent local ones (agreement 1.5e-08, cross-rank
disagreement 0.0, local gradients differing by 4.6e-01 so the check cannot pass vacuously).

**Validation is split by scene, and the number is much worse for it.** There are ~10 crops per
camera frame and consecutive frames show the same objects, so a record-level split puts
near-duplicate crops of the same car on both sides. Measured directly: under a random split
validation accuracy reaches 0.72–0.76 and val loss tracks train loss; under a scene-level split
the same model gets 0.27–0.65 and val loss has no trend at all. The first number is the one a
careless write-up would report. Training loss falls monotonically (1.14 → 0.23 over 10 epochs)
either way — the loop is correct; the model simply does not generalise across scenes yet.

**Run `make doctor` before anything else.** On a host that cannot resolve its own hostname, bare
`torchrun` never completes rendezvous — it retries a failing lookup with exponential backoff and
never times out into a useful error. On the development machine this also added ~55s to every
run, because torch's elastic agent calls `socket.getfqdn()` once per worker lifecycle event to
label a telemetry field, and each call cost a 5s resolver timeout. Eleven calls, 55.1s, on a job
whose actual distributed work takes 29ms. `make hello` works around it with
`--local-addr=127.0.0.1` and `GLOO_SOCKET_IFNAME=lo0`; `doctor` prints the real `/etc/hosts` fix.

**nuScenes ships 3D boxes, not 2D.** Producing image crops means projecting each annotation
through `world → ego → camera → pixels`. That math is written by hand in `src/dtp/geometry.py`
rather than taken from `nuscenes-devkit`, which wants `numpy<2` and pulls in matplotlib, OpenCV,
scikit-learn and shapely — too much to sit underneath a training loop. To keep the hand-rolled
version honest, `tests/test_geometry.py` compares it against a fixture the devkit generated in a
throwaway environment: 881 boxes, agreeing to **5e-05 m** on camera-frame corners and **5e-05 px**
on projected boxes, which is the rounding floor of the fixture itself. The devkit is not a
dependency of this project.

## Layout

```
src/dtp/
  dist.py             process group setup/teardown, rank-tagged logging
  hello.py            4-process gloo verification (make hello)
  doctor.py           host preflight: hostname resolution, start method (make doctor)
  probe.py            how DataLoader workers are started and seeded (make probe)
  manifest.py         nuScenes sample enumeration (copied from the sibling repo)
  nuscenes_tables.py  minimal JSON metadata reader, no devkit
  geometry.py         hand-rolled 3D → 2D projection
  build_crops.py      crop manifest builder (make crops)
  dataset.py          map-style Dataset: (tensor, label)
tools/
  gen_oracle_fixture.py   regenerates the devkit reference fixture (separate venv)
```

Methodology and benchmark numbers will live in `MEASUREMENT.md`, separate from this file, once
milestone B produces them.
