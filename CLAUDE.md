# CLAUDE.md

Working instructions for this repo. Read this at the start of every session.

## What this project is

A distributed training pipeline for nuScenes camera crops, built from the primitives up.
Milestones A and B are **pure PyTorch DDP with no Ray, deliberately** — the point is to
understand what higher-level frameworks abstract before adopting them. Ray Train (and
possibly FSDP) land later as additional backends over the same data and training layer.

The owner is an ML platform engineer with deep Spark/Databricks experience and **no prior
production distributed-*training* or GPU-cluster experience**. This repo closes that gap
with first-hand measurements instead of hedged answers. The goal is understanding the
material, not shipping a demo.

Sibling repo: [`ray-multimodal-pipeline`](https://github.com/netwrkerror/ray-multimodal-pipeline)
— same nuScenes data, but *inference* (Ray Data + Faster R-CNN batch inference). This repo is
*training*. They are siblings, not a merge. `manifest.py` is copied over from it with a comment
noting its origin; **do not build a shared package** — two personal repos don't justify it.

## Working agreement

- **One coherent feature per session.** Do not scaffold everything at once. Do not start the
  next milestone item because the current one finished early — stop and let it be reviewed.
- **The user does all commits, PRs, and merges.** Propose changes; never run `git commit`,
  `git push`, `gh pr create`, or `git merge`.
- **Explain the why.** For every feature, state the underlying systems concept and what breaks
  without it. This repo is a learning vehicle. An unexplained working implementation is a
  failed session.
- **Tests matter.** Several features below are only meaningful if an assertion proves the
  property holds. A feature whose defining assertion is missing is not done.
- At the end of each completed feature, append a section to `NOTES.md` (see its own header for
  the required structure). **`NOTES.md` is gitignored and stays local** — it is the user's
  personal learning log, not a repo deliverable. Still write it: it is part of "done", and
  the fact that it is untracked is not a reason to skip or shorten it.

## Branch, commit, and PR proposals

The user lands all git operations. Claude's job is to hand over ready-to-use text so nothing has
to be composed by hand.

**At the start of a feature**, before writing code, propose the branch name and state which
milestone item it implements:

```
branch: feat/a5-distributed-sampler
implements: A5 — DistributedSampler and the no-duplicate assertion
```

**At the end of a feature**, once the "done when" condition is met, propose all four artifacts
together in one block: branch name, commit message, PR title, PR description.

Naming and format:

- **Branch:** `<type>/<milestone-id>-<short-slug>`, lowercase and hyphenated —
  `feat/a6-atomic-checkpointing`, `bench/b1-dataloader-harness`, `docs/a4-notes-entry`,
  `fix/b5-worker-seeding`. Types in use: `feat`, `fix`, `bench`, `docs`, `chore`.
- **Commit message:** Conventional Commits. Subject line `<type>(<milestone-id>): <imperative
  summary>`, at most 72 characters, no trailing period. Then a blank line, then a body that says
  *what changed and why it is correct* — for this repo that means naming the property the tests
  prove, and any measured number. Body lines wrap at 72. One commit per feature unless the work
  genuinely splits (e.g. reproducing a bug, then fixing it — those are two commits, and the
  reproduction commit is worth keeping in the history).
- **PR title:** the commit subject line, minus the conventional-commit prefix, capitalized —
  `A6: atomic checkpointing with crash-safe writes`.
- **PR description:** four short sections — *What* (one or two sentences), *Why* (the systems
  concept, one paragraph — this is the same idea as the NOTES.md entry, compressed), *How to
  verify* (the exact commands to run, including the test that proves the defining property), and
  *Notes* (anything deferred, anything measured, anything still uncertain). Do **not** link
  `NOTES.md` from a PR — it is untracked, so the link would 404; carry the one-paragraph
  version of its *Why* into the PR body instead.

Templates, including the required trailers:

```
feat(a6): write checkpoints atomically with a latest.json marker

Rank 0 writes to a temp file, fsyncs, then os.replace()s into position, so
a crash mid-write leaves the previous checkpoint intact rather than a
truncated file that torch.load happily accepts. Other ranks wait on a
barrier. Saves model.module.state_dict() to avoid the module. key prefix.

Test simulates a crash between write and replace and asserts the prior
checkpoint still loads.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

````markdown
## What
<one or two sentences>

## Why
<the systems concept, one paragraph; what breaks without this>

## How to verify
```bash
<exact commands, including the defining test>
```

## Notes
<deferred work, measured numbers, open questions>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
````

Never run `git commit`, `git push`, `git merge`, or `gh pr create` — propose the text and stop.

## Environment

- macOS on Apple Silicon, VS Code + Claude extension.
- **CPU-only for Milestones A and B.** `gloo` backend. No CUDA, no NCCL.
- Launch with `torchrun` (e.g. `torchrun --nproc_per_node=4 ...`), not manual process spawning.
- DataLoader worker start method (fork vs spawn) on Apple Silicon is a known trap — verify it
  empirically rather than assuming the Linux default.

## Conventions

- Python 3.11+
- `uv` for dependency management (`pyproject.toml` + `uv.lock`, both committed)
- `ruff` for lint and format
- `pytest` for tests
- `MEASUREMENT.md` is **separate from the README** and holds methodology: how each number was
  produced, the hardware it ran on, warmup and discard policy, repeat count, and the noise floor.
- A task runner for common commands (train, test, lint, bench) so measurements are one command.

## Measurement standard

This is the bar the sibling repo set and it holds here:

- Report **medians across repeats**, not single runs. Warm up and discard the first N batches.
- State the **noise floor** before comparing configs.
- If two configurations are statistically indistinguishable inside that noise floor, **say so
  explicitly**. Never claim a win the data doesn't support, and never round a wash into a
  headline. "No measurable difference" is a valid and valuable result.
- Every number in a write-up must be reproducible from a committed command.

## The seven gotchas

Each is a real bug that neither crashes nor logs. **Reproduce each one before fixing it** —
observing the failure is part of the deliverable, not overhead.

- [ ] 1. Missing `set_epoch()` → identical shuffle every epoch (A5)
- [ ] 2. Uneven split with `drop_last=False` → hang at the collective (A8)
- [ ] 3. Forked workers sharing NumPy seed → identical augmentations across workers (B5)
- [ ] 4. `IterableDataset` sharded by rank but not by worker → silently duplicated data (B5)
- [ ] 5. Saving the DDP wrapper's `state_dict` → every key prefixed `module.`, load fails (A6)
- [ ] 6. Non-atomic checkpoint write → corrupt file the resume path loads happily (A6)
- [ ] 7. `loss.item()` inside the loop → forced synchronization (matters more on GPU;
      instrument it now) (A3/A4)

## Phases at a glance

| Phase | Milestone | Theme | Items | Runs on |
|---|---|---|---|---|
| 1 | A | DDP mechanics — correctness | A1–A5 | CPU, gloo, 4 procs |
| 2 | A | DDP mechanics — durability | A6–A8 | CPU, gloo, 4 procs |
| 3 | B | Measure the input pipeline | B1–B3 | CPU, single proc |
| 4 | B | Fix it, prove the fix | B4–B6 | CPU, gloo |
| — | C | GPU scaling | later | rented multi-GPU |
| — | D | Ray Train backend | later | rented multi-GPU |

Phases 1–4 are strictly sequential; items within a phase are too. Phase 2 depends on a training
loop that already converges (A3) and a sampler that already partitions correctly (A5), because
otherwise a checkpoint bug and a data bug look the same. Phase 4 depends on phase 3's baseline
numbers existing first — a before/after with no *before* is not a result.

## Milestones and status

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done

### Milestone A — DDP mechanics on CPU

A correct, resumable, multi-process training job, no GPU involved.

- [ ] **A1 — Repo scaffold and dev environment.** Package layout, `pyproject.toml` + `uv.lock`,
  ruff, pytest, task runner. Confirm `torch.distributed`/`gloo` works across 4 processes on
  macOS *before anything else*; verify DataLoader worker start method.
  *Done when:* `torchrun --nproc_per_node=4` runs a hello-world printing rank/world_size from
  every process and exits cleanly.
- [ ] **A2 — Dataset layer: nuScenes crops.** `jsonl` manifest of object crops + class labels
  from the mini split. Manifest stays decoupled from loading code so B can swap storage layout.
  *Done when:* a map-style `Dataset` yields `(tensor, label)`, with tests for a missing file and
  a malformed record.
- [ ] **A3 — Single-process training baseline.** ResNet-18-scale or smaller, plain loop,
  structured logging, per-epoch metrics. **Must converge before any DDP code exists** —
  otherwise a distributed bug and a modeling bug are indistinguishable.
  *Done when:* loss decreases over a few epochs and step time is recorded.
- [ ] **A4 — DDP wrapper.** `init_process_group("gloo")`, rank/local_rank/world_size from
  `torchrun` env, `DistributedDataParallel`, rank-0-only logging, clean `destroy_process_group`
  on exit *and on exception*.
  *Done when:* a 4-process run completes, per-rank losses agree closely, no hang on exit.
  *Concepts:* why gradients not weights are synchronized; gradient bucketing and
  backward/comm overlap; effective batch size = `per_device × world_size` and the LR
  implication; what changes for `nccl`.
- [ ] **A5 — DistributedSampler + no-duplicate assertion.** `set_epoch()` wired up. Gather the
  indices every rank sees in an epoch; **assert the union covers the dataset with no overlap**,
  and that order changes between epochs. Then break it deliberately (gotcha 1).
  *Done when:* both assertions pass and the "every rank must shuffle *identically* before
  striding" argument can be stated from memory.
- [ ] **A6 — Checkpointing.** Rank-0-only writes, barrier for the others. Atomic:
  temp file → `fsync` → `os.replace`. Save `model.module.state_dict()`, optimizer state, epoch,
  global step, sampler epoch. Keep last N plus best.
  *Done when:* `latest.json` points at a valid checkpoint, and a test simulating a crash
  mid-write leaves the previous checkpoint intact and loadable.
- [ ] **A7 — Resume.** Correct epoch, step count, and sampler epoch, so data order doesn't
  silently restart.
  *Done when:* a run killed at epoch 3 and resumed produces a loss curve continuous with an
  uninterrupted run.
- [ ] **A8 — Fault injection and recovery.** Kill a worker mid-run; **observe the failure
  honestly first** (surviving ranks block at the next collective until timeout). Diagnose with
  `py-spy dump` per process. Then add `torchrun --max-restarts` with c10d rendezvous restarting
  from the last checkpoint. Also reproduce the uneven-split hang (gotcha 2), then fix it.
  *Done when:* both hang signatures can be described from having seen them, and the job recovers
  from a killed worker.

### Milestone B — Measure and fix the input pipeline

The highest-value milestone: the exact problem an AV training team lives with.

- [ ] **B1 — DataLoader-only throughput harness.** Iterate the loader with **no model at all**,
  measure records/sec, sweep `num_workers` ∈ {0, 1, 2, 4, 8}. Warm up, discard first N batches,
  median across repeats, emit CSV.
  *Done when:* one reproducible command produces a throughput-vs-workers table.
- [ ] **B2 — Find and explain the knee.** Where does adding workers stop helping? Correlate
  against physical core count, per-record decode time, and whether the ceiling is CPU or file
  I/O. **The explanation is the deliverable, not the chart.**
- [ ] **B3 — `persistent_workers` on short epochs.** Measure worker startup cost with and
  without; express it as a percentage of epoch time.
- [ ] **B4 — Shard repacker.** Thousands of small files → large sequential shards
  (WebDataset-style tar vs. simple binary format — discuss the trade-off first). Target a few
  hundred MB per shard.
  *Done when:* a CLI repacks a directory into shards with a manifest and content round-trips
  identically.
- [ ] **B5 — Iterable dataset over shards.** `IterableDataset` reading shards sequentially with
  a shuffle buffer. **Shard twice — by rank and by worker id** (gotcha 4). Test no record is
  duplicated across the rank × worker grid. Fix the seeding trap with `worker_init_fn` and
  prove augmentations differ across workers (gotcha 3).
- [ ] **B6 — Before/after benchmark and write-up.** Rerun B1 against the sharded layout.
  Comparison table, charts, and `MEASUREMENT.md` explaining what bottlenecked and why.
  **Three paragraphs of analysis is worth more than the code.**

### Later — not now

Milestone C (rented multi-GPU: scaling table at 1/2/4 GPUs, `nvidia-smi dmon` starvation
traces, mixed precision, one `torch.profiler` trace) and Milestone D (Ray Train `TorchTrainer`
+ `ScalingConfig`, evaluate Ray Data for decoupled CPU preprocessing). Possibly FSDP as a
second backend. **The sequencing is the point: DDP first, framework second.**

## Out of scope for A and B

CUDA, NCCL, multi-node, FSDP, mixed precision, Ray Train. All deferred deliberately. If a
proposed change requires any of them, it belongs in a later milestone.
