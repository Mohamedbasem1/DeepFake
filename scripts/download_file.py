from __future__ import annotations

import argparse
from pathlib import Path

import requests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download a file with redirects and size checks.")
    parser.add_argument("--url", required=True, help="Direct download URL.")
    parser.add_argument("--output", required=True, type=Path, help="Output file path.")
    parser.add_argument(
        "--cookies",
        type=Path,
        help="Optional Netscape cookies.txt file for authenticated downloads.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    if args.cookies:
        session.cookies.update(load_netscape_cookies(args.cookies))

    with session.get(args.url, stream=True, allow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" in content_type:
            raise RuntimeError(
                "The server returned HTML, not the dataset file. "
                "This usually means the URL needs login cookies or is not a direct download link."
            )

        total = int(response.headers.get("content-length") or 0)
        written = 0
        with args.output.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                written += len(chunk)
                if total:
                    percent = 100.0 * written / total
                    print(f"\r{written / 1e9:.2f} / {total / 1e9:.2f} GB ({percent:.1f}%)", end="")

    print(f"\nSaved {args.output} ({written / 1e9:.2f} GB)")


def load_netscape_cookies(path: Path) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 7:
            cookies[parts[5]] = parts[6]
    return cookies


if __name__ == "__main__":
    main()

