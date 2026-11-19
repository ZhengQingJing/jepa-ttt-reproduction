"""Install the v1.0.0 reproduction assets after verifying their repository-pinned SHA-256."""

import hashlib
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    "https://github.com/ZhengQingJing/jepa-ttt-reproduction/releases/download/v1.0.0/"
)
DESTINATIONS = {
    "offline.npz": ROOT / "data/offline.npz",
    "scratch-weights.pt": ROOT / "models/scratch-weights.pt",
    "scratch-reward.pt": ROOT / "models/scratch-reward.pt",
}


def checksum(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    hashes = dict()
    for line in (ROOT / "assets.sha256").read_text().splitlines():
        if line.strip():
            digest, name = line.split()
            hashes[name.lstrip("*")] = digest
    if set(DESTINATIONS) != set(hashes):
        raise ValueError("assets.sha256 must contain exactly the three release assets")
    for name, target in DESTINATIONS.items():
        digest = hashes[name]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if checksum(target) != digest:
                raise SystemExit(
                    f"Existing {target} has a different hash. Move it before installing the release."
                )
            print(f"Verified {name}")
            continue
        with tempfile.TemporaryDirectory(dir=target.parent) as temp:
            partial = Path(temp) / name
            request = urllib.request.Request(
                RELEASE + name, headers={"User-Agent": "jepa-ttt-reproduction/1.0.0"}
            )
            with (
                urllib.request.urlopen(request, timeout=90) as source,
                partial.open("wb") as dest,
            ):
                while block := source.read(1 << 20):
                    dest.write(block)
            if checksum(partial) != digest:
                raise ValueError(f"Checksum mismatch for {name}")
            partial.replace(target)
        print(f"Installed {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
