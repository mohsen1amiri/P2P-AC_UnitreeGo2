#!/usr/bin/env python3
"""Local copy of the NVIDIA simulation assets used by the Go2 task (robot model, ground, sky, textures).

When Isaac Sim starts, it loads the Unitree Go2 model and a few textures from NVIDIA's asset server. This tool
downloads exactly those files once into ``assets/isaac_sim/`` so that runs need no internet access, are not
affected by changes on the server, and use exactly the same files every time. ``scripts/container.sh`` then
points Isaac Sim to the local copy automatically (see ``p2pac/isaac_assets.py``).

    python3 scripts/setup/isaac_assets.py download     # download and verify (plain Python 3, no packages needed)
    python3 scripts/setup/isaac_assets.py check        # verify an existing copy (sizes and SHA-256)

The list of files with their sizes and SHA-256 checksums is ``assets/isaac_sim_manifest.json`` (committed to Git).
It was computed by following every reference inside the files (``build-manifest``, needs Pixar's ``usd-core``
package; only needed if the task starts using other assets).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIRROR = os.path.join(REPO, "assets", "isaac_sim")
MANIFEST = os.path.join(REPO, "assets", "isaac_sim_manifest.json")
BASE_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1"

# Assets referenced by the Go2 task (paths relative to the asset root; see the module docstring of
# p2pac/isaac_assets.py for where each one is used).
ROOTS = [
    "Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd",                      # robot (isaaclab_assets/robots/unitree.py)
    "Isaac/Environments/Grid/default_environment.usd",                # flat ground (GroundPlaneCfg)
    "Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",   # sky light
    "Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    "Isaac/Props/UIElements/arrow_x.usd",                             # command arrows (only when recording videos)
]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(rel: str, dest_root: str = MIRROR, base_url: str = BASE_URL, timeout: float = 120.0) -> str:
    """Download ``base_url/rel`` to ``dest_root/rel`` (atomic). Returns the local path."""
    dest = os.path.join(dest_root, *rel.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    url = f"{base_url}/{urllib.request.quote(rel)}"
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    return dest


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------
def cmd_download(args) -> int:
    man = load_manifest()
    files = man["files"]
    total = sum(f["bytes"] for f in files)
    print(f"Downloading {len(files)} files ({total / 1e6:.1f} MB) from {man['base_url']}")
    print(f"into {MIRROR}")
    for f in files:
        dest = os.path.join(MIRROR, *f["path"].split("/"))
        if os.path.isfile(dest) and os.path.getsize(dest) == f["bytes"] and sha256(dest) == f["sha256"]:
            print(f"  ok        {f['path']}")
            continue
        try:
            fetch(f["path"], base_url=man["base_url"])
        except (urllib.error.URLError, OSError) as e:
            print(f"  FAILED    {f['path']}: {e}")
            return 1
        ok = sha256(dest) == f["sha256"]
        print(f"  {'download' if ok else 'MISMATCH'}  {f['path']}")
        if not ok:
            print("The file on the server differs from the one the experiments used (checksum mismatch).")
            return 1
    print("All files present and verified.")
    return 0


def cmd_check(args) -> int:
    man = load_manifest()
    bad = 0
    for f in man["files"]:
        dest = os.path.join(MIRROR, *f["path"].split("/"))
        if not os.path.isfile(dest):
            print(f"  missing   {f['path']}")
            bad += 1
        elif os.path.getsize(dest) != f["bytes"] or sha256(dest) != f["sha256"]:
            print(f"  CHANGED   {f['path']}")
            bad += 1
    print("complete and verified" if not bad else f"{bad} file(s) missing or changed: run 'download'")
    return 0 if not bad else 1


# ----------------------------------------------------------------------------------------------------
_MDL_TEXTURE = re.compile(r'texture_2d\(\s*"([^"]+)"')


def _dependencies(local: str) -> list[str]:
    """Asset paths referenced by one file, as written in it (USD: sublayers, references, payloads, asset-valued
    attributes such as textures and materials; MDL: textures)."""
    if local.endswith(".mdl"):
        with open(local, encoding="utf-8", errors="replace") as f:
            return _MDL_TEXTURE.findall(f.read())
    if not local.endswith((".usd", ".usda", ".usdc")):
        return []
    from pxr import Sdf, UsdUtils  # maintainer-only dependency (pip install usd-core)

    layer = Sdf.Layer.FindOrOpen(local)
    out: list[str] = []
    UsdUtils.ModifyAssetPaths(layer, lambda p: (out.append(p), p)[1])   # visits every asset path in the layer
    return out


def _resolve(ref: str, parent_rel: str) -> str | None:
    """Map a reference found in ``parent_rel`` to a path relative to the asset root (None: not downloadable)."""
    ref = ref.strip()
    if not ref or "<UDIM>" in ref or ref.startswith(("omniverse://", "file:")):
        return None
    if ref.startswith("http"):
        if not ref.startswith(BASE_URL + "/"):
            return None
        return ref[len(BASE_URL) + 1:]
    if ref.startswith("/"):
        return None
    if not (ref.startswith("./") or ref.startswith("../") or "/" in ref or "." in os.path.basename(ref)):
        return None
    parts = parent_rel.split("/")[:-1]
    for p in ref.split("/"):
        if p in ("", "."):
            continue
        if p == "..":
            if parts:
                parts.pop()
        else:
            parts.append(p)
    return "/".join(parts)


def cmd_build_manifest(args) -> int:
    seen: dict[str, dict] = {}
    skipped: dict[str, str] = {}
    queue = list(ROOTS)
    while queue:
        rel = queue.pop(0)
        if rel in seen or rel in skipped:
            continue
        try:
            local = fetch(rel)
        except urllib.error.HTTPError as e:
            skipped[rel] = f"HTTP {e.code} (not on the asset server: a built-in Isaac Sim resource)"
            continue
        seen[rel] = {"path": rel, "bytes": os.path.getsize(local), "sha256": sha256(local)}
        for ref in _dependencies(local):
            r = _resolve(ref, rel)
            if r is None:
                skipped.setdefault(ref, "not a file on the asset server (built-in or absolute path)")
            elif r not in seen:
                queue.append(r)
        print(f"  {seen[rel]['bytes']:>10d}  {rel}")
    man = {
        "description": "NVIDIA Isaac Sim 5.1 assets loaded by the Go2 navigation task (scripts/setup/isaac_assets.py)",
        "base_url": BASE_URL,
        "roots": ROOTS,
        "files": sorted(seen.values(), key=lambda f: f["path"]),
        "not_downloaded": skipped,
    }
    with open(MANIFEST, "w") as f:
        json.dump(man, f, indent=2)
    print(f"{len(seen)} files, {sum(f['bytes'] for f in seen.values()) / 1e6:.1f} MB -> {MANIFEST}")
    for k, v in skipped.items():
        print(f"  not downloaded: {k}  ({v})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("download", "check", "build-manifest"))
    args = ap.parse_args()
    return {"download": cmd_download, "check": cmd_check, "build-manifest": cmd_build_manifest}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
