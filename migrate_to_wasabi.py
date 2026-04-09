"""
migrate_to_wasabi.py
====================
Migration of all local data to Wasabi S3.

Bucket layout:
    sierra-geo-harvester  — raw geodata (LiDAR, EMIT, CSVs, shapefiles, OSM, DEMs)
    cell-data             — processed outputs (DuckDB, GeoJSON exports, agent results)
    logs                  — logs, error logs (created automatically if missing)

Safe to run multiple times — skips files already uploaded.

Usage:
    python migrate_to_wasabi.py              # full migration
    python migrate_to_wasabi.py --dry-run    # show what would happen
    python migrate_to_wasabi.py --db-only    # just sync the DB to cell-data
    python migrate_to_wasabi.py --check      # show all three buckets
    python migrate_to_wasabi.py --include lidar  # only upload lidar
"""

import os
import sys
import argparse
import time
from pathlib import Path

ENDPOINT = "https://s3.us-east-1.wasabisys.com"
REGION   = "us-east-1"

# (local_path, bucket, s3_prefix, description)
UPLOAD_MAPS = [
    # ── sierra-geo-harvester: raw geodata ─────────────────────────────────────
    (Path("/mnt/c/Geodata/Fault"),
     "sierra-geo-harvester", "fault",       "Fault shapefiles"),

    (Path("/mnt/c/Geodata/Textual"),
     "sierra-geo-harvester", "textual",     "CSV/textual data"),

    (Path("/mnt/c/Geodata/remote_sensing"),
     "sierra-geo-harvester", "dem",         "DEM GeoTIFFs"),

    (Path("/mnt/c/Geodata/hydrology"),
     "sierra-geo-harvester", "hydrology",   "Hydrology data"),

    (Path("/mnt/c/Geodata/OSM (California & Nevada)"),
     "sierra-geo-harvester", "osm",         "OSM PBF files"),

    (Path("/mnt/c/Geodata/lidar/Sierra25LasFiles"),
     "sierra-geo-harvester", "lidar/Sierra25LasFiles", "LiDAR zips (~50GB)"),

    (Path("/home/placer/sierra_prospector/data/hyperspectral"),
     "sierra-geo-harvester", "hyperspectral", "EMIT hyperspectral"),

    # ── cell-data: processed outputs ──────────────────────────────────────────
    (Path("/home/placer/sierra_prospector/db"),
     "cell-data", "db",      "DuckDB database"),

    (Path("/home/placer/sierra_prospector/outputs"),
     "cell-data", "outputs", "GeoJSON exports and agent outputs"),

    # ── logs: log files ───────────────────────────────────────────────────────
    (Path("/home/placer/sierra_prospector/logs"),
     "logs", "",             "Pipeline logs and alert logs"),
]

SKIP_EXTENSIONS = {".tmp", ".pyc", ".bak"}
SKIP_NAMES      = {".gitkeep", "desktop.ini", "thumbs.db"}


def get_client():
    try:
        import boto3
        from botocore.config import Config
        access_key = os.environ.get("WASABI_ACCESS_KEY")
        secret_key = os.environ.get("WASABI_SECRET_KEY")
        if not access_key or not secret_key:
            print("ERROR: WASABI_ACCESS_KEY and WASABI_SECRET_KEY not set.")
            print("Add them to ~/.bashrc and run: source ~/.bashrc")
            sys.exit(1)
        return boto3.client(
            "s3",
            endpoint_url          = ENDPOINT,
            aws_access_key_id     = access_key,
            aws_secret_access_key = secret_key,
            region_name           = REGION,
            config                = Config(retries={"max_attempts": 3}),
        )
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)


def ensure_bucket(client, bucket: str):
    """Create bucket if it does not exist."""
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(Bucket=bucket)
            print(f"  Created bucket: {bucket}")
        except Exception as e:
            print(f"  Could not create bucket {bucket}: {e}")


def get_existing_keys(client, bucket: str) -> set:
    keys = set()
    try:
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                keys.add(obj["Key"])
    except Exception as e:
        print(f"  Warning: could not list {bucket}: {e}")
    return keys


def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def upload_directory(client, local_dir: Path, bucket: str, s3_prefix: str,
                     existing: set, dry_run: bool) -> dict:
    stats = {"uploaded": 0, "skipped": 0, "failed": 0,
             "bytes_uploaded": 0, "bytes_skipped": 0}

    if not local_dir.exists():
        print(f"  Not found: {local_dir} — skipping")
        return stats

    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    print(f"  {len(files)} files in {local_dir.name}")

    for f in files:
        if f.suffix.lower() in SKIP_EXTENSIONS:
            continue
        if f.name.lower() in SKIP_NAMES:
            continue

        relative = f.relative_to(local_dir)
        s3_key   = f"{s3_prefix}/{relative}".replace("\\", "/").lstrip("/")
        size     = f.stat().st_size

        if s3_key in existing:
            stats["skipped"]       += 1
            stats["bytes_skipped"] += size
            continue

        if dry_run:
            print(f"  [DRY RUN] s3://{bucket}/{s3_key} ({human_size(size)})")
            stats["uploaded"]       += 1
            stats["bytes_uploaded"] += size
            continue

        try:
            print(f"  ↑ {f.name} ({human_size(size)})...", end=" ", flush=True)
            t0 = time.time()
            client.upload_file(str(f), bucket, s3_key)
            elapsed = round(time.time() - t0, 1)
            speed   = human_size(int(size / max(elapsed, 0.1)))
            print(f"done ({elapsed}s, {speed}/s)")
            stats["uploaded"]       += 1
            stats["bytes_uploaded"] += size
        except Exception as e:
            print(f"FAILED: {e}")
            stats["failed"] += 1

    return stats


def sync_db_only(client, dry_run: bool):
    """Quick sync of just the DuckDB to cell-data bucket."""
    db_path = Path("/home/placer/sierra_prospector/db/prospector.duckdb")
    if not db_path.exists():
        print("DB not found")
        return
    s3_key = "db/prospector.duckdb"
    size   = db_path.stat().st_size
    print(f"Syncing DB to cell-data: {human_size(size)}")
    if dry_run:
        print(f"  [DRY RUN] s3://cell-data/{s3_key}")
        return
    try:
        client.upload_file(str(db_path), "cell-data", s3_key)
        print(f"  Done: s3://cell-data/{s3_key}")
    except Exception as e:
        print(f"  Failed: {e}")


def check_wasabi(client):
    """Print summary of all three buckets."""
    for bucket in ["sierra-geo-harvester", "cell-data", "logs"]:
        print(f"\ns3://{bucket}")
        print("-" * 50)
        try:
            paginator = client.get_paginator("list_objects_v2")
            by_prefix = {}
            total = 0
            for page in paginator.paginate(Bucket=bucket):
                for obj in page.get("Contents", []):
                    prefix = obj["Key"].split("/")[0] or "(root)"
                    by_prefix[prefix] = by_prefix.get(prefix, 0) + obj["Size"]
                    total += obj["Size"]
            if not by_prefix:
                print("  (empty)")
            else:
                for prefix, size in sorted(by_prefix.items()):
                    print(f"  {prefix:<35} {human_size(size):>10}")
                print(f"  {'TOTAL':<35} {human_size(total):>10}")
        except Exception as e:
            print(f"  Error or bucket does not exist: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true")
    parser.add_argument("--db-only",  action="store_true")
    parser.add_argument("--check",    action="store_true")
    parser.add_argument("--include",  type=str, default=None,
                        help="Only upload paths containing this string")
    args = parser.parse_args()

    client = get_client()

    if args.check:
        check_wasabi(client)
        return

    if args.db_only:
        sync_db_only(client, args.dry_run)
        return

    # Ensure all buckets exist (creates logs if missing)
    print("Checking buckets...")
    for bucket in ["sierra-geo-harvester", "cell-data", "logs"]:
        ensure_bucket(client, bucket)

    print(f"\nMigrating data to Wasabi")
    if args.dry_run:
        print("DRY RUN — nothing will be uploaded")
    print("=" * 60)

    # Build per-bucket existing key sets
    existing_by_bucket = {}

    totals = {"uploaded": 0, "skipped": 0, "failed": 0,
              "bytes_uploaded": 0, "bytes_skipped": 0}

    for local_dir, bucket, s3_prefix, description in UPLOAD_MAPS:
        if args.include and args.include.lower() not in str(local_dir).lower():
            continue

        # Load existing keys for this bucket (cached)
        if bucket not in existing_by_bucket:
            print(f"\nScanning s3://{bucket}...")
            existing_by_bucket[bucket] = get_existing_keys(client, bucket)
            print(f"  {len(existing_by_bucket[bucket])} files already there")

        print(f"\n[{description}]")
        print(f"  {local_dir}")
        print(f"  -> s3://{bucket}/{s3_prefix}")

        stats = upload_directory(
            client, local_dir, bucket, s3_prefix,
            existing_by_bucket[bucket], args.dry_run
        )

        for k, v in stats.items():
            totals[k] += v

        print(f"  Result: {stats['uploaded']} uploaded, "
              f"{stats['skipped']} skipped, "
              f"{stats['failed']} failed")

    print(f"\n{'='*60}")
    print(f"MIGRATION COMPLETE")
    print(f"  Uploaded: {totals['uploaded']} files ({human_size(totals['bytes_uploaded'])})")
    print(f"  Skipped:  {totals['skipped']} already on Wasabi ({human_size(totals['bytes_skipped'])})")
    print(f"  Failed:   {totals['failed']} files")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
