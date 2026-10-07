#!/usr/bin/env python3
"""OPTIONAL: download the Python packages of requirements/isaac-overlay.txt as wheel files, so that
scripts/setup/install_overlay.sh can install them on an HPC without internet access.

    python3 scripts/setup/download_wheels.py --arch x86_64     # -> container/wheelhouse/x86_64/   (e.g. Alvis)
    python3 scripts/setup/download_wheels.py --arch aarch64    # -> container/wheelhouse/aarch64/  (e.g. Arrhenius GPU)

Run it on any computer with internet access and Python 3 with pip (a laptop is fine): the packages are downloaded
for the container's platform (Linux, Python 3.11, the chosen CPU architecture), whatever system you run this on.
Packages published only as source code (pure Python) are turned into platform-independent wheels. Lines of the
requirements file with a platform marker (e.g. ``; platform_machine == "x86_64"``) are skipped for other
architectures. Then copy the repository folder, including container/wheelhouse/, to the HPC. The wheelhouse is not
stored in Git (one x86_64 package is larger than GitHub's 100 MB file limit).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REQ = os.path.join(REPO, "requirements", "isaac-overlay.txt")
PLATFORMS = {
    "x86_64": ["manylinux_2_28_x86_64", "manylinux_2_17_x86_64", "manylinux2014_x86_64", "manylinux1_x86_64",
               "linux_x86_64"],
    # Ubuntu 22.04 (the base of the Isaac Sim 5.1 image) has glibc 2.35, so manylinux_2_34 wheels are usable.
    "aarch64": ["manylinux_2_34_aarch64", "manylinux_2_28_aarch64", "manylinux_2_17_aarch64", "manylinux2014_aarch64",
                "linux_aarch64"],
}


def requirements(arch: str) -> list[str]:
    from pip._vendor.packaging.markers import Marker  # pip's bundled copy of 'packaging' (no extra install)

    out = []
    for line in open(REQ):
        line = line.split("#")[0].strip()
        if not line:
            continue
        spec, _, marker = line.partition(";")
        if marker.strip() and not Marker(marker.strip()).evaluate({"platform_machine": arch, "sys_platform": "linux",
                                                                     "python_version": "3.11"}):
            print(f"  skip (not for {arch}): {spec.strip()}")
            continue
        out.append(spec.strip())
    return out


def pip(*args: str) -> int:
    return subprocess.call([sys.executable, "-m", "pip", *args])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", required=True, choices=("x86_64", "aarch64"))
    ap.add_argument("--out", default=None, help="default: container/wheelhouse/<arch>")
    args = ap.parse_args()
    out = args.out or os.path.join(REPO, "container", "wheelhouse", args.arch)
    os.makedirs(out, exist_ok=True)
    plat = [a for p in PLATFORMS[args.arch] for a in ("--platform", p)]
    binary = ["--only-binary=:all:", *plat, "--python-version", "3.11", "--implementation", "cp",
              "--abi", "cp311", "--abi", "abi3", "--abi", "none"]
    failed = []
    for spec in requirements(args.arch):
        if pip("download", "-q", "--no-deps", *binary, "-d", out, spec) == 0:
            continue
        print(f"  building a wheel for {spec} (no binary wheel; pure Python source)")
        if pip("wheel", "-q", "--no-deps", "-w", out, spec) != 0:
            failed.append(spec)
    n = len([f for f in os.listdir(out) if f.endswith(".whl")])
    print(f"{n} wheels in {out}")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
