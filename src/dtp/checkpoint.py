"""Crash-safe checkpointing.

A checkpoint write is not one operation. It is allocate, write bytes, flush, close -
and a crash can land between any two of them, leaving a file that exists, has a
plausible size, and is garbage. The naive `torch.save(state, "latest.pt")` overwrites
the only good copy in place, so a crash during the write destroys the checkpoint you
were relying on to recover from crashes.

The fix is that `os.replace` is atomic on POSIX: a reader sees either the old inode or
the new one, never a mixture. So:

    write to <name>.tmp  ->  fsync the file  ->  os.replace into place  ->  fsync the dir

`fsync` on the file forces bytes out of the page cache onto the device; without it the
rename can be durable while the contents are not. The `fsync` on the *directory* is
the step most implementations skip: the rename is a directory metadata change, and it
can be lost in a power failure even when the file data survived.

Two nuances specific to distributed training:

  * Only rank 0 writes, and the other ranks wait at a barrier. Four processes writing
    the same path race, and a rank that runs ahead to the next epoch while rank 0 is
    still writing will desynchronise the next collective.
  * `model.state_dict()` on a DDP-wrapped model prefixes every key with `module.`,
    because the wrapper is itself a Module holding the real one as a submodule. Saved
    that way, the checkpoint will not load into an unwrapped model - which is exactly
    what a single-process resume or an inference script uses.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

LATEST = "latest.json"
BEST = "best.json"


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the real model, unwrapping DistributedDataParallel if present.

    Gotcha 5. `DistributedDataParallel` is a Module whose only child is the model you
    handed it, so its `state_dict()` keys all read `module.conv1.weight` rather than
    `conv1.weight`. Saving that and loading it into a plain model fails with a wall of
    "unexpected key" errors, and the usual response - `strict=False` - silently loads
    *nothing* and leaves you with random weights that train from scratch while
    appearing to have resumed.
    """
    return model.module if hasattr(model, "module") else model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save(obj: Any, path: Path) -> str:
    """torch.save via temp file + fsync + os.replace. Returns the file's sha256.

    The temp file is created in the *destination directory* on purpose: os.replace is
    only atomic within a single filesystem, so a temp file in /tmp could land on a
    different device and silently degrade to a copy.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh:
            torch.save(obj, fh)
            fh.flush()
            os.fsync(fh.fileno())  # bytes are on the device, not just in the page cache

        digest = _sha256(tmp)
        os.replace(tmp, path)  # atomic: readers see the old file or the new one

        # The rename is directory metadata and needs its own fsync to be durable.
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
        return digest
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class CheckpointMeta:
    """What the marker file records about a checkpoint."""

    filename: str
    epoch: int
    global_step: int
    sampler_epoch: int
    val_loss: float
    sha256: str
    world_size: int


class CheckpointManager:
    """Rank-0 writes, everyone waits, nothing is overwritten in place."""

    def __init__(self, directory: str | Path, keep_last: int = 3, is_master: bool = True) -> None:
        self.dir = Path(directory)
        self.keep_last = keep_last
        self.is_master = is_master

    # -- writing ---------------------------------------------------------------

    def save(
        self,
        *,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        global_step: int,
        sampler_epoch: int,
        val_loss: float,
        extra: dict | None = None,
    ) -> CheckpointMeta | None:
        """Write a checkpoint from rank 0; other ranks wait for it to finish.

        Returns the metadata on rank 0 and None elsewhere.
        """
        meta: CheckpointMeta | None = None
        if self.is_master:
            self.dir.mkdir(parents=True, exist_ok=True)
            state = {
                # unwrapped: see unwrap_model. Saving the DDP wrapper's keys is gotcha 5.
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                # Saved so a resumed run continues the data order rather than
                # restarting it. Without this the sampler reshuffles from epoch 0 and
                # the resumed run re-sees data it already trained on.
                "sampler_epoch": sampler_epoch,
                "val_loss": val_loss,
                **(extra or {}),
            }
            filename = f"ckpt_epoch{epoch:04d}.pt"
            digest = atomic_save(state, self.dir / filename)

            meta = CheckpointMeta(
                filename=filename,
                epoch=epoch,
                global_step=global_step,
                sampler_epoch=sampler_epoch,
                val_loss=float(val_loss),
                sha256=digest,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
            )
            # The marker is written *after* the checkpoint is durable, so it can never
            # point at a file that does not exist or is half-written.
            atomic_write_json(asdict(meta), self.dir / LATEST)

            best = self.read_marker(BEST)
            if best is None or val_loss < best.val_loss:
                atomic_write_json(asdict(meta), self.dir / BEST)

            self._prune()

        # Ranks that did not write must not run ahead into the next collective while
        # rank 0 is still on the filesystem.
        if dist.is_initialized():
            dist.barrier()
        return meta

    def _prune(self) -> None:
        """Keep the newest `keep_last` checkpoints, plus whatever `best` points at."""
        keep = set()
        for marker in (LATEST, BEST):
            meta = self.read_marker(marker)
            if meta is not None:
                keep.add(meta.filename)

        checkpoints = sorted(self.dir.glob("ckpt_epoch*.pt"))
        for path in checkpoints[: max(0, len(checkpoints) - self.keep_last)]:
            if path.name not in keep:
                path.unlink(missing_ok=True)

    # -- reading ---------------------------------------------------------------

    def read_marker(self, name: str = LATEST) -> CheckpointMeta | None:
        path = self.dir / name
        if not path.exists():
            return None
        try:
            with open(path) as fh:
                return CheckpointMeta(**json.load(fh))
        except (json.JSONDecodeError, TypeError):
            return None

    def load(self, name: str = LATEST, verify: bool = True) -> dict | None:
        """Load the checkpoint the marker points at, verifying its checksum.

        Gotcha 6 says a corrupt checkpoint gets loaded happily by the resume path. The
        checksum is what makes that impossible here: a truncated or altered file is
        rejected loudly instead of resuming from garbage.
        """
        meta = self.read_marker(name)
        if meta is None:
            return None

        path = self.dir / meta.filename
        if not path.exists():
            raise FileNotFoundError(f"{name} points at {meta.filename}, which does not exist")

        if verify:
            actual = _sha256(path)
            if actual != meta.sha256:
                raise ValueError(
                    f"{meta.filename} is corrupt: sha256 {actual[:12]} does not match the "
                    f"{meta.sha256[:12]} recorded in {name}"
                )

        return torch.load(path, map_location="cpu", weights_only=False)
