"""Download the trained YOLOv8m weights into models/best.pt.

Usage:
    python scripts/download_weights.py [--url URL] [--output models/best.pt]
"""
import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_URL = (
    "https://github.com/gabrielyannarek04-max/military-object-detection/"
    "releases/download/v1.0.0/best.pt"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Weights download URL")
    parser.add_argument("--output", default="models/best.pt", help="Destination path")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.url}")
    try:
        urllib.request.urlretrieve(args.url, out)
    except urllib.error.URLError as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1

    print(f"Saved to {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
