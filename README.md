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
| A3 | Single-process training baseline | ⬜ |
| A4 | DDP wrapper | ⬜ |
| A5 | DistributedSampler and the no-duplicate assertion | ⬜ |
| A6 | Checkpointing | ⬜ |
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
