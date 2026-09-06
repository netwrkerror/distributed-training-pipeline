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
| A7 | Resume | ✅ |
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
make resume     # continue from the latest checkpoint
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

One line each; the reasoning behind them is in the merged PR descriptions.

- **Run `make doctor` first.** On a host that cannot resolve its own hostname, bare `torchrun`
  never completes rendezvous, and torch's elastic agent adds ~55s per run calling
  `socket.getfqdn()` to label a telemetry field — 11 calls at 5.007s, on a job whose distributed
  work takes 29ms.
- **Validation splits by scene, not by record.** With ~10 crops per frame, a record-level split
  puts near-duplicate crops of the same car on both sides: 0.72–0.76 val accuracy against
  0.27–0.65 for an honest split. The first number is the one a careless write-up reports.
- **Most of DDP's apparent cost was a launcher default.** 120.3ms → 284.6ms per step looks like a
  2.4× penalty; 1.9× of it is `torchrun` setting `OMP_NUM_THREADS=1`, and only 1.25× is gradient
  synchronisation.
- **"Per-rank losses agree" does not prove gradients are synchronised** — with identical data on
  every rank it holds even with the all-reduce deleted. `make check-ddp` asserts the real
  property against ranks given deliberately different data.
- **A corrupt checkpoint usually loads without complaint.** Scanning one flipped byte across a
  checkpoint, 12 of 19 positions loaded silently with finite, plausibly scaled weights; truncation
  is caught by torch's zip container, bit corruption is not. Checkpoints here carry a sha256.
- **DDP synchronises gradients, not state.** BatchNorm running statistics diverge 7.07e-03 across
  ranks even with `broadcast_buffers=True`, so a rank-0 checkpoint is an incomplete snapshot of a
  distributed run — which is why a DDP resume is continuous but not bit-exact.

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
