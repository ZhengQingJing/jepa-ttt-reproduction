"""Check out the exact public LeWM and simulator revisions used by this reproduction."""

from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    "le-wm": (
        "https://github.com/lucas-maes/le-wm.git",
        "8edfeb336732b5f3ce7b8b210d0ba370a09e2cac",
    ),
    "stable-worldmodel": (
        "https://github.com/galilai-group/stable-worldmodel.git",
        "21446f1ede6d5284e981bd7b47f432b994e6d812",
    ),
}


def git(directory, *args, capture=False):
    return subprocess.run(
        ["git", "-C", str(directory), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def main():
    parent = ROOT / "upstream"
    parent.mkdir(exist_ok=True)
    for name, (url, revision) in REPOSITORIES.items():
        target = parent / name
        if target.exists():
            actual = git(target, "rev-parse", "HEAD", capture=True).strip()
            if actual != revision:
                raise SystemExit(f"{target} is at {actual}; expected {revision}.")
            print(f"{name}: {revision}")
            continue
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            checkout = Path(temp) / name
            checkout.mkdir()
            git(checkout, "init", "--quiet")
            git(checkout, "remote", "add", "origin", url)
            git(checkout, "fetch", "--depth=1", "origin", revision)
            git(checkout, "checkout", "--quiet", "--detach", "FETCH_HEAD")
            actual = git(checkout, "rev-parse", "HEAD", capture=True).strip()
            if actual != revision:
                raise RuntimeError(f"Expected {revision}; received {actual}.")
            checkout.rename(target)
        print(f"{name}: {revision}")


if __name__ == "__main__":
    main()
