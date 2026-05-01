from __future__ import annotations

import argparse
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib.parse import quote

import msal
import requests


GRAPH = "https://graph.microsoft.com/v1.0"
SCOPES = ["Files.ReadWrite.All", "offline_access"]
CHUNK_SIZE = 10 * 1024 * 1024


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a local folder recursively to OneDrive.")
    parser.add_argument("--source", required=True, type=Path, help="Local folder to upload, e.g. data")
    parser.add_argument("--remote", required=True, help="OneDrive remote folder, e.g. ImageCLEF_Backup/data")
    parser.add_argument("--client-id", default=os.environ.get("MS_GRAPH_CLIENT_ID"), help="Microsoft app client ID")
    parser.add_argument("--tenant", default=os.environ.get("MS_GRAPH_TENANT", "consumers"))
    parser.add_argument("--token-cache", type=Path, default=Path(".msal_token_cache.bin"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.client_id:
        raise SystemExit(
            "Missing --client-id. Create a Microsoft app registration and pass its Application/Client ID, "
            "or set MS_GRAPH_CLIENT_ID."
        )
    if not args.source.is_dir():
        raise SystemExit(f"Source folder does not exist: {args.source}")

    files = [path for path in sorted(args.source.rglob("*")) if path.is_file()]
    print(f"files: {len(files)}")
    print(f"source: {args.source}")
    print(f"remote: {args.remote}")
    if args.dry_run:
        for path in files[:20]:
            print(path)
        print("dry run only")
        return

    token = acquire_token(args.client_id, args.tenant, args.token_cache)
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

    ensure_remote_folder(session, args.remote)
    for index, path in enumerate(files, start=1):
        rel = path.relative_to(args.source).as_posix()
        remote_path = "/".join(part.strip("/") for part in [args.remote, rel] if part)
        size = path.stat().st_size
        print(f"[{index}/{len(files)}] {rel} ({size / 1e6:.1f} MB)")
        if args.skip_existing and remote_exists(session, remote_path):
            print("  exists, skipping")
            continue
        upload_file(session, path, remote_path)


def acquire_token(client_id: str, tenant: str, cache_path: Path) -> str:
    cache = msal.SerializableTokenCache()
    if cache_path.exists():
        cache.deserialize(cache_path.read_text(encoding="utf-8"))

    app = msal.PublicClientApplication(
        client_id=client_id,
        authority=f"https://login.microsoftonline.com/{tenant}",
        token_cache=cache,
    )
    accounts = app.get_accounts()
    result = app.acquire_token_silent(SCOPES, account=accounts[0] if accounts else None)
    if not result:
        flow = app.initiate_device_flow(scopes=SCOPES)
        if "user_code" not in flow:
            raise RuntimeError(f"Could not create device flow: {flow}")
        print(flow["message"])
        result = app.acquire_token_by_device_flow(flow)

    if cache.has_state_changed:
        cache_path.write_text(cache.serialize(), encoding="utf-8")

    if "access_token" not in result:
        raise RuntimeError(json.dumps(result, indent=2))
    return result["access_token"]


def ensure_remote_folder(session: requests.Session, remote_folder: str) -> None:
    current = ""
    for part in [part for part in remote_folder.strip("/").split("/") if part]:
        parent_path = current
        current = "/".join([current, part]).strip("/")
        if remote_exists(session, current):
            continue
        if parent_path:
            url = f"{GRAPH}/me/drive/root:/{escape_path(parent_path)}:/children"
        else:
            url = f"{GRAPH}/me/drive/root/children"
        response = session.post(
            url,
            json={"name": part, "folder": {}, "@microsoft.graph.conflictBehavior": "replace"},
        )
        response.raise_for_status()


def remote_exists(session: requests.Session, remote_path: str) -> bool:
    response = session.get(f"{GRAPH}/me/drive/root:/{escape_path(remote_path)}")
    if response.status_code == 404:
        return False
    response.raise_for_status()
    return True


def upload_file(session: requests.Session, local_path: Path, remote_path: str) -> None:
    size = local_path.stat().st_size
    parent = "/".join(remote_path.split("/")[:-1])
    if parent:
        ensure_remote_folder(session, parent)

    if size <= 4 * 1024 * 1024:
        simple_upload(session, local_path, remote_path)
    else:
        large_upload(session, local_path, remote_path)


def simple_upload(session: requests.Session, local_path: Path, remote_path: str) -> None:
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    headers = {"Content-Type": content_type}
    with local_path.open("rb") as handle:
        response = session.put(
            f"{GRAPH}/me/drive/root:/{escape_path(remote_path)}:/content",
            data=handle,
            headers=headers,
        )
    response.raise_for_status()


def large_upload(session: requests.Session, local_path: Path, remote_path: str) -> None:
    create_response = session.post(
        f"{GRAPH}/me/drive/root:/{escape_path(remote_path)}:/createUploadSession",
        json={"item": {"@microsoft.graph.conflictBehavior": "replace", "name": local_path.name}},
    )
    create_response.raise_for_status()
    upload_url = create_response.json()["uploadUrl"]

    size = local_path.stat().st_size
    with local_path.open("rb") as handle:
        start = 0
        while start < size:
            chunk = handle.read(CHUNK_SIZE)
            end = start + len(chunk) - 1
            headers = {
                "Content-Length": str(len(chunk)),
                "Content-Range": f"bytes {start}-{end}/{size}",
            }
            for attempt in range(5):
                response = requests.put(upload_url, headers=headers, data=chunk, timeout=120)
                if response.status_code in {200, 201, 202}:
                    break
                if response.status_code in {429, 500, 502, 503, 504}:
                    sleep_for = 2 ** attempt
                    print(f"  retry {attempt + 1} after status {response.status_code}, sleeping {sleep_for}s")
                    time.sleep(sleep_for)
                    continue
                response.raise_for_status()
            else:
                response.raise_for_status()
            start = end + 1
            print(f"  {100 * start / size:.1f}%")


def escape_path(path: str) -> str:
    return "/".join(quote(part) for part in path.strip("/").split("/"))


if __name__ == "__main__":
    main()

