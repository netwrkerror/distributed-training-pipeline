"""Environment preflight: the machine-level defects that make torchrun hang or crawl.

Run with:  make doctor

Nothing here is about this repo's code. These are properties of the host that
silently ruin a distributed run, and each one cost real debugging time in A1:

  * If the host cannot resolve its own hostname, every socket.getfqdn() call blocks
    for the full DNS timeout. torch's elastic agent calls it once per worker
    lifecycle event purely to label telemetry, so an unresolvable hostname taxes a
    job that does no work at all.
  * gloo separately resolves the hostname to pick an interface to bind to, and
    falls back to loopback only after the same timeout.

The fix is to make the name resolve; the environment variables and flags this repo
passes are a workaround, not a cure.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import platform
import socket
import time

SLOW_LOOKUP_S = 0.5


def _timed(fn, *args):
    start = time.perf_counter()
    try:
        return fn(*args), None, time.perf_counter() - start
    except Exception as exc:  # any resolver failure is what we are here to report
        return None, exc, time.perf_counter() - start


def main() -> int:
    hostname = socket.gethostname()
    print(f"platform        : {platform.platform()}")
    print(f"python          : {platform.python_version()}")
    print(f"cpu count       : {os.cpu_count()}")
    print(f"mp start method : {mp.get_start_method()}  (available: {mp.get_all_start_methods()})")
    print(f"hostname        : {hostname}")

    problems: list[str] = []

    fqdn, _, fqdn_s = _timed(socket.getfqdn, hostname)
    print(f"getfqdn()       : {fqdn!r} in {fqdn_s:.3f}s")
    if fqdn_s > SLOW_LOOKUP_S:
        problems.append(
            f"socket.getfqdn() took {fqdn_s:.3f}s. torch's elastic agent calls this once per "
            f"worker lifecycle event, so a 4-process job pays it ~11 times."
        )

    for family, label in ((socket.AF_INET, "IPv4"), (socket.AF_INET6, "IPv6")):
        res, exc, dt = _timed(socket.getaddrinfo, hostname, 0, family)
        if exc is not None:
            print(f"resolve {label}    : FAILED in {dt:.3f}s -> {exc}")
            problems.append(f"{label} resolution of {hostname!r} fails after {dt:.3f}s.")
        else:
            print(f"resolve {label}    : {res[0][4]} in {dt:.3f}s")

    print()
    if not problems:
        print("OK: no host-level problems detected.")
        return 0

    print("PROBLEMS FOUND")
    for p in problems:
        print(f"  - {p}")
    print(
        "\nLikely fix (requires sudo, and repairs this for every tool on the machine,\n"
        "not just torch):\n"
        f"    echo '127.0.0.1 {hostname}' | sudo tee -a /etc/hosts\n"
        f"    echo '::1 {hostname}' | sudo tee -a /etc/hosts\n"
        "\nUntil then this repo works around it by passing --local-addr=127.0.0.1 to\n"
        "torchrun and setting GLOO_SOCKET_IFNAME=lo0. See the Makefile."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
