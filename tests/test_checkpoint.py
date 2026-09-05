"""A6: checkpoints must survive a crash during the write.

The failure this guards against is specific and nasty: the naive
`torch.save(state, "latest.pt")` overwrites the only good copy in place, so a crash
partway through destroys the checkpoint you were keeping in order to survive crashes.
The file still exists afterwards and has a plausible size.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from dtp.checkpoint import (
    BEST,
    LATEST,
    CheckpointManager,
    atomic_save,
    unwrap_model,
)


class FakeDDP(nn.Module):
    """Mimics DistributedDataParallel's shape: a Module wrapping the real one.

    Real DDP needs a process group, but the property under test is purely about
    `state_dict()` key naming, which this reproduces exactly.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def _manager(tmp_path: Path, **kwargs) -> CheckpointManager:
    return CheckpointManager(tmp_path / "checkpoints", **kwargs)


def _save(manager: CheckpointManager, model, optimizer, epoch: int, val_loss: float):
    return manager.save(
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        global_step=epoch * 10,
        sampler_epoch=epoch,
        val_loss=val_loss,
    )


# --- the marker contract ------------------------------------------------------


def test_latest_marker_points_at_a_loadable_checkpoint(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    manager = _manager(tmp_path)
    _save(manager, model, optimizer, epoch=0, val_loss=1.0)

    meta = manager.read_marker(LATEST)
    assert meta is not None
    assert (manager.dir / meta.filename).exists()

    state = manager.load()
    assert state["epoch"] == 0
    assert state["global_step"] == 0
    assert state["sampler_epoch"] == 0
    assert set(state["model"]) == set(model.state_dict())


def test_state_round_trips_into_a_fresh_model(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    # take a step so the optimizer has real state to restore
    model(torch.randn(2, 4)).sum().backward()
    optimizer.step()

    manager = _manager(tmp_path)
    _save(manager, model, optimizer, epoch=2, val_loss=0.5)
    state = manager.load()

    restored = _model()
    restored.load_state_dict(state["model"])
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        assert torch.equal(a, b)

    fresh_opt = torch.optim.AdamW(restored.parameters(), lr=0.1)
    fresh_opt.load_state_dict(state["optimizer"])
    assert fresh_opt.state_dict()["state"], "optimizer state did not survive"


# --- gotcha 5: the module. prefix ---------------------------------------------


def test_ddp_wrapper_keys_are_stripped_before_saving(tmp_path: Path) -> None:
    """Gotcha 5. Saved keys must be the plain model's, not the wrapper's."""
    plain = _model()
    wrapped = FakeDDP(plain)
    assert all(k.startswith("module.") for k in wrapped.state_dict())

    manager = _manager(tmp_path)
    _save(manager, wrapped, torch.optim.AdamW(wrapped.parameters()), epoch=0, val_loss=1.0)
    saved = manager.load()["model"]

    assert not any(k.startswith("module.") for k in saved)
    # the real test: it loads into an unwrapped model with strict=True
    _model().load_state_dict(saved)


def test_saving_the_wrapper_state_dict_is_what_breaks(tmp_path: Path) -> None:
    """The bug, demonstrated, so the fix above is not taken on faith.

    Note the second half: `strict=False` is the usual reflex, and it 'succeeds' while
    loading nothing at all - the model keeps its random initialisation and trains from
    scratch while the logs say it resumed.
    """
    wrapped = FakeDDP(_model())
    path = tmp_path / "wrong.pt"
    atomic_save(wrapped.state_dict(), path)
    loaded = torch.load(path, weights_only=False)

    target = _model()
    with pytest.raises(RuntimeError, match=r"Unexpected key|Missing key"):
        target.load_state_dict(loaded)

    before = [p.detach().clone() for p in target.parameters()]
    result = target.load_state_dict(loaded, strict=False)
    assert result.missing_keys, "strict=False hid a total failure to load"
    for p, q in zip(target.parameters(), before, strict=True):
        assert torch.equal(p.detach(), q), "expected strict=False to load nothing"


def test_unwrap_model_is_identity_for_plain_models() -> None:
    model = _model()
    assert unwrap_model(model) is model
    assert unwrap_model(FakeDDP(model)) is model


# --- gotcha 6 and the A6 done-when: crash mid-write ---------------------------


def test_crash_during_write_leaves_the_previous_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The A6 done-when.

    Epoch 0 is written successfully. The write for epoch 1 dies at the moment of
    `os.replace` - the worst possible instant, with a complete temp file on disk and
    the rename not yet done. Afterwards the epoch-0 checkpoint must still load, and
    the marker must still point at it.
    """
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    manager = _manager(tmp_path)
    _save(manager, model, optimizer, epoch=0, val_loss=1.0)

    good = manager.load()
    assert good["epoch"] == 0

    def boom(*args, **kwargs):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr("dtp.checkpoint.os.replace", boom)
    with pytest.raises(OSError, match="simulated crash"):
        _save(manager, model, optimizer, epoch=1, val_loss=0.5)
    monkeypatch.undo()

    # the previous checkpoint is untouched and still loads
    recovered = manager.load()
    assert recovered["epoch"] == 0
    assert manager.read_marker(LATEST).epoch == 0

    # and no half-written file was left lying around under a real name
    assert sorted(p.name for p in manager.dir.glob("ckpt_epoch*.pt")) == ["ckpt_epoch0000.pt"]


def test_no_temp_files_survive_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    model = _model()
    _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=0, val_loss=1.0)

    monkeypatch.setattr(
        "dtp.checkpoint.os.replace", lambda *a, **k: (_ for _ in ()).throw(OSError("boom"))
    )
    with pytest.raises(OSError):
        _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=1, val_loss=0.1)
    monkeypatch.undo()

    leftovers = [p.name for p in manager.dir.iterdir() if ".tmp" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"


@pytest.mark.parametrize("position", [0.5, 0.9])
def test_corrupt_checkpoint_is_rejected_not_resumed_from(tmp_path: Path, position: float) -> None:
    """Gotcha 6, reproduced and then made impossible.

    The premise of the gotcha is that a corrupt checkpoint is loaded *happily* by the
    resume path, and it holds. torch.save writes a zip container, so a truncated file
    is caught by the format itself - but a flipped byte is not, at 0.5 (metadata
    region) or at 0.9 (inside the tensor payload). In the second case the weights come
    back finite and plausibly scaled, so training resumes from silently wrong
    parameters and simply gets worse for no visible reason.

    The `verify=False` half of this test is not decoration: it is what demonstrates
    that the checksum is doing the catching, rather than torch.load having noticed.
    """
    model = _model()
    manager = _manager(tmp_path)
    _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=0, val_loss=1.0)

    target = manager.dir / manager.read_marker(LATEST).filename
    data = bytearray(target.read_bytes())
    data[int(len(data) * position)] ^= 0xFF
    target.write_bytes(bytes(data))

    # unguarded, this is the bug: the corrupt file loads without complaint
    smuggled = manager.load(verify=False)
    assert smuggled["epoch"] == 0, "expected torch.load to accept the corrupt file"

    # guarded, it is caught
    with pytest.raises(ValueError, match="corrupt"):
        manager.load()


def test_truncated_checkpoint_is_caught_by_the_container(tmp_path: Path) -> None:
    """The one corruption torch.load does catch on its own, recorded for contrast."""
    model = _model()
    manager = _manager(tmp_path)
    _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=0, val_loss=1.0)

    target = manager.dir / manager.read_marker(LATEST).filename
    target.write_bytes(target.read_bytes()[: int(target.stat().st_size * 0.6)])

    with pytest.raises((RuntimeError, EOFError, ValueError)):
        manager.load(verify=False)


def test_marker_pointing_at_a_missing_file_raises(tmp_path: Path) -> None:
    model = _model()
    manager = _manager(tmp_path)
    _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=0, val_loss=1.0)
    (manager.dir / manager.read_marker(LATEST).filename).unlink()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        manager.load()


def test_unreadable_marker_is_treated_as_absent(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.dir.mkdir(parents=True)
    (manager.dir / LATEST).write_text("{ truncated json")
    assert manager.read_marker(LATEST) is None
    assert manager.load() is None


# --- retention ----------------------------------------------------------------


def test_keeps_last_n_plus_best(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters())
    manager = _manager(tmp_path, keep_last=2)

    # epoch 0 is the best (lowest val_loss) and must survive being aged out
    losses = [0.1, 0.9, 0.8, 0.7, 0.6]
    for epoch, loss in enumerate(losses):
        _save(manager, model, optimizer, epoch=epoch, val_loss=loss)

    remaining = sorted(p.name for p in manager.dir.glob("ckpt_epoch*.pt"))
    assert "ckpt_epoch0000.pt" in remaining, "best checkpoint was pruned"
    assert "ckpt_epoch0004.pt" in remaining, "latest checkpoint was pruned"
    assert len(remaining) <= 3, remaining

    assert manager.read_marker(BEST).epoch == 0
    assert manager.read_marker(LATEST).epoch == 4
    assert manager.load(BEST)["epoch"] == 0


def test_non_master_ranks_do_not_write(tmp_path: Path) -> None:
    model = _model()
    manager = _manager(tmp_path, is_master=False)
    meta = _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=0, val_loss=1.0)
    assert meta is None
    assert not (manager.dir / LATEST).exists()


def test_marker_records_provenance(tmp_path: Path) -> None:
    model = _model()
    manager = _manager(tmp_path)
    _save(manager, model, torch.optim.AdamW(model.parameters()), epoch=3, val_loss=0.25)
    payload = json.loads((manager.dir / LATEST).read_text())
    assert payload["epoch"] == 3
    assert payload["global_step"] == 30
    assert payload["sampler_epoch"] == 3
    assert payload["world_size"] == 1
    assert len(payload["sha256"]) == 64


def test_atomic_save_uses_the_destination_directory_for_temps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """os.replace is only atomic within one filesystem, so the temp file must not be
    somewhere like /tmp that could be a different device."""
    seen = {}
    real = os.replace

    def spy(src, dst):
        seen["src_parent"] = Path(src).parent
        seen["dst_parent"] = Path(dst).parent
        return real(src, dst)

    monkeypatch.setattr("dtp.checkpoint.os.replace", spy)
    atomic_save({"x": 1}, tmp_path / "nested" / "ckpt.pt")
    assert seen["src_parent"] == seen["dst_parent"]
