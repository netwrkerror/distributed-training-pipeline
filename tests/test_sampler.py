"""A5: DistributedSampler partitioning, asserted rather than assumed.

The feature is only worth doing because of these assertions. Sharding that silently
overlaps trains some records twice per epoch and others never; sharding that silently
drops records shrinks the dataset. Neither crashes, neither logs, and both look like
a model that "just doesn't quite converge".

None of this needs four processes. `DistributedSampler` takes `num_replicas` and
`rank` explicitly, so a single test process can construct all four ranks' samplers and
inspect the whole partition at once - which is strictly better than a distributed test,
because it can compare the ranks against each other instead of each rank checking only
itself.
"""

from __future__ import annotations

import pytest
from torch.utils.data import DistributedSampler

WORLD = 4


class _Fake:
    """Length is the only thing DistributedSampler asks of a dataset."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


def _indices(n: int, world: int = WORLD, epoch: int = 0, **kwargs) -> list[list[int]]:
    """The indices each rank would see this epoch."""
    out = []
    for rank in range(world):
        sampler = DistributedSampler(_Fake(n), num_replicas=world, rank=rank, **kwargs)
        sampler.set_epoch(epoch)
        out.append(list(sampler))
    return out


def test_ranks_partition_the_dataset_exactly() -> None:
    """The assertion that makes this feature worth doing.

    Every record goes to exactly one rank: the union covers the dataset and the ranks
    do not overlap.
    """
    n = 3208  # the real training split size
    per_rank = _indices(n)
    union = [i for shard in per_rank for i in shard]

    assert len(union) == n, f"expected {n} indices in total, got {len(union)}"
    assert sorted(union) == list(range(n)), "union is not exactly the dataset"
    assert len(set(union)) == len(union), "some index went to more than one rank"


def test_every_rank_gets_the_same_number_of_batches() -> None:
    """Unequal shards mean one rank runs more steps than the others and blocks at the
    next collective waiting for peers that have already finished the epoch. This is
    the uneven-split hang A8 reproduces deliberately."""
    per_rank = _indices(3208)
    sizes = {len(shard) for shard in per_rank}
    assert len(sizes) == 1, f"ranks got different shard sizes: {[len(s) for s in per_rank]}"


def test_all_ranks_shuffle_identically_before_striding() -> None:
    """The mechanism, stated as a test.

    Partitioning happens with no communication whatsoever. Each rank independently
    produces the *same* permutation - it is seeded by `seed + epoch`, which every rank
    knows - and then takes `permutation[rank::world_size]`. Interleaving the ranks'
    shards in rank order must therefore reconstruct that shared permutation exactly.

    If the ranks shuffled differently, their strides would overlap and miss records,
    and nothing would notice, because no rank ever sees another rank's indices.
    """
    n = 3208
    per_rank = _indices(n, epoch=3)
    interleaved = [idx for group in zip(*per_rank, strict=True) for idx in group]

    assert sorted(interleaved) == list(range(n))
    # a real shuffle, not the identity
    assert interleaved != list(range(n))


def test_order_changes_between_epochs_when_set_epoch_is_called() -> None:
    first = _indices(3208, epoch=0)
    second = _indices(3208, epoch=1)
    assert first != second, "shuffle did not change between epochs"
    # ...but the partition is still exact in both
    for shards in (first, second):
        assert sorted(i for s in shards for i in s) == list(range(3208))


def test_without_set_epoch_every_epoch_is_identical() -> None:
    """Gotcha 1, as an assertion rather than a warning.

    A sampler whose `set_epoch` is never called keeps epoch 0 forever, so every epoch
    replays the same permutation in the same order. The job runs, the loss goes down,
    nothing is logged and nothing crashes - the model simply sees far less variety
    than the code appears to provide.
    """
    epochs = []
    for epoch in range(3):
        sampler = DistributedSampler(_Fake(3208), num_replicas=WORLD, rank=0)
        # deliberately NOT calling set_epoch(epoch)
        del epoch
        epochs.append(list(sampler))

    assert epochs[0] == epochs[1] == epochs[2], "expected the frozen-shuffle bug"

    # and the contrast: the same sampler with set_epoch does vary
    varied = [_indices(3208, epoch=e)[0] for e in range(3)]
    assert len({tuple(v) for v in varied}) == 3


@pytest.mark.parametrize("n", [3207, 3209, 10, 7])
def test_padding_duplicates_exactly_the_shortfall(n: int) -> None:
    """When the dataset does not divide evenly, `drop_last=False` **pads**.

    This is the detail that makes a naive "no duplicates" assertion wrong. To keep
    every rank at an equal number of steps, DistributedSampler repeats records from
    the front of the permutation. Every record is still seen at least once, but
    `world_size - (n % world_size)` of them are seen twice in the same epoch.
    """
    per_rank = _indices(n)
    union = [i for shard in per_rank for i in shard]
    shortfall = (-n) % WORLD

    assert len(union) == n + shortfall
    assert set(union) == set(range(n)), "padding must not lose records"
    assert len(union) - len(set(union)) == shortfall
    assert len({len(s) for s in per_rank}) == 1


@pytest.mark.parametrize("n", [3207, 3209, 10])
def test_drop_last_trades_coverage_for_no_duplicates(n: int) -> None:
    """The other side of the trade: `drop_last=True` never duplicates, but the tail
    of the dataset is not seen at all this epoch."""
    per_rank = _indices(n, drop_last=True)
    union = [i for shard in per_rank for i in shard]

    assert len(union) == (n // WORLD) * WORLD
    assert len(set(union)) == len(union), "drop_last must not duplicate"
    assert len(set(range(n)) - set(union)) == n % WORLD


def test_dropped_records_differ_between_epochs() -> None:
    """Mitigates the obvious objection to drop_last: with reshuffling, the records
    dropped in one epoch are (very likely) seen in the next, so nothing is
    systematically excluded from training."""
    n = 3209
    dropped = []
    for epoch in range(4):
        union = {i for s in _indices(n, epoch=epoch, drop_last=True) for i in s}
        dropped.append(frozenset(set(range(n)) - union))
    assert len(set(dropped)) > 1, "the same records are dropped every epoch"
