.PHONY: install lint fmt test test-all hello probe doctor crops train check

NPROC ?= 4

# nuScenes mini lives outside this repo (9GB); point at it rather than copying.
DTP_NUSCENES_ROOT ?= /Users/nabh/workspace/repos/github/ray-multimodal-pipeline/data/nuscenes
CROPS             ?= data/crops.jsonl
EPOCHS            ?= 10

# Workarounds for a host that cannot resolve its own hostname; see `make doctor`.
# --local-addr stops the elastic agent advertising an unresolvable FQDN to its workers.
# GLOO_SOCKET_IFNAME stops gloo paying a DNS timeout before falling back to loopback.
GLOO_ENV       ?= GLOO_SOCKET_IFNAME=lo0
TORCHRUN_FLAGS ?= --master-addr=127.0.0.1 --local-addr=127.0.0.1

install:            ## create the venv and install everything
	uv sync

lint:               ## ruff check
	uv run ruff check .

fmt:                ## ruff format + import sort
	uv run ruff format .
	uv run ruff check --fix .

test:               ## fast tests only
	uv run pytest -m "not slow"

test-all:           ## every test, including the ones that spawn real processes
	uv run pytest

hello:              ## A1 done-when: 4-process gloo hello-world
	$(GLOO_ENV) uv run torchrun $(TORCHRUN_FLAGS) --nproc_per_node=$(NPROC) -m dtp.hello

crops:              ## build the crop manifest from nuScenes (run once)
	uv run python -m dtp.build_crops --dataroot $(DTP_NUSCENES_ROOT) --out $(CROPS)

train:              ## A3 done-when: single-process baseline (scene-level split)
	uv run python -m dtp.train --epochs $(EPOCHS)

probe:              ## report how DataLoader workers are started on this machine
	uv run python -m dtp.probe

doctor:             ## check the host for defects that make distributed runs hang
	uv run python -m dtp.doctor

check: lint test    ## what CI would run
