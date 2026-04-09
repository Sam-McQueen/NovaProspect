"""
sierra_prospector/core/storage.py
=====================================
Transparent cloud storage manager for Wasabi S3.

Bucket layout:
    sierra-geo-harvester  — raw geodata
    cell-data             — DB, GeoJSON, processed outputs
    logs                  — pipeline logs

Agents never touch boto3 directly:
    storage.get_geodata("lidar/Sierra25LasFiles/Sierra25_360.zip")
    storage.get_celldata("db/prospector.duckdb")
    storage.put_celldata(local_path, "db/prospector.duckdb")
    storage.put_log(local_path, "alerts.log")
"""

import os
import threading
import time
from pathlib import Path
from typing import Optional, List

from core.logger import get_logger

log = get_logger("storage")

BUCKET_GEODATA   = "sierra-geo-harvester"
BUCKET_CELLDATA  = "cell-data"
BUCKET_LOGS      = "logs"
WASABI_REGION    = "us-east-1"
WASABI_ENDPOINT  = f"https://s3.{WASABI_REGION}.wasabisys.com"

CACHE_DIR = Path("/home/placer/sierra_prospector/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_CACHE_BYTES = 50 * 1024 ** 3   # 50GB


class StorageManager:
    """Transparent Wasabi S3 access. Thread-safe."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._client  = None
        self._enabled = False
        self._init_client()

    def _init_client(self):
        try:
            import boto3
            from botocore.config import Config
            access_key = os.environ.get("WASABI_ACCESS_KEY")
            secret_key = os.environ.get("WASABI_SECRET_KEY")
            if not access_key or not secret_key:
                log.warning(
                    "Wasabi credentials not found in environment. "
                    "Falling back to local paths only."
                )
                return
            self._client = boto3.client(
                "s3",
                endpoint_url          = WASABI_ENDPOINT,
                aws_access_key_id     = access_key,
                aws_secret_access_key = secret_key,
                region_name           = WASABI_REGION,
                config                = Config(
                    retries={"max_attempts": 3, "mode": "standard"},
                    max_pool_connections=10,
                ),
            )
            self._enabled = True
            log.info("Wasabi storage initialised", endpoint=WASABI_ENDPOINT)
        except ImportError:
            log.warning("boto3 not installed — pip install boto3")
        except Exception as e:
            log.warning("Wasabi init failed", error=str(e))

    @property
    def enabled(self) -> bool:
        return self._enabled

    # ── Get (download) ────────────────────────────────────────────────────────

    def get_geodata(self, s3_key: str, force: bool = False) -> Optional[Path]:
        """Get a raw geodata file from sierra-geo-harvester."""
        return self._get(BUCKET_GEODATA, s3_key, force)

    def get_celldata(self, s3_key: str, force: bool = False) -> Optional[Path]:
        """Get a processed output file from cell-data."""
        return self._get(BUCKET_CELLDATA, s3_key, force)

    def _get(self, bucket: str, s3_key: str, force: bool) -> Optional[Path]:
        local_path = CACHE_DIR / bucket / s3_key
        if local_path.exists() and not force:
            return local_path
        if not self._enabled:
            return self._local_fallback(s3_key)
        return self._download(bucket, s3_key, local_path)

    def _download(self, bucket: str, s3_key: str, local_path: Path) -> Optional[Path]:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        try:
            log.info("Downloading", bucket=bucket, key=s3_key)
            t0 = time.time()
            self._client.download_file(bucket, s3_key, str(tmp))
            tmp.rename(local_path)
            elapsed = round(time.time() - t0, 1)
            size_mb = local_path.stat().st_size / 1024 / 1024
            log.info("Download complete",
                     key=s3_key, size_mb=round(size_mb, 1), elapsed_s=elapsed)
            return local_path
        except Exception as e:
            log.error("Download failed", bucket=bucket, key=s3_key, error=str(e))
            if tmp.exists():
                tmp.unlink()
            return None

    def _local_fallback(self, s3_key: str) -> Optional[Path]:
        """Try legacy local Windows/WSL paths if cloud not available."""
        for base in [Path("/mnt/c/Geodata"),
                     Path("/home/placer/sierra_prospector/data")]:
            p = base / s3_key
            if p.exists():
                return p
        return None

    # ── Put (upload) ──────────────────────────────────────────────────────────

    def put_celldata(self, local_path: Path, s3_key: str) -> bool:
        """Upload a processed output to cell-data bucket."""
        return self._put(BUCKET_CELLDATA, local_path, s3_key)

    def put_log(self, local_path: Path, s3_key: str = None) -> bool:
        """Upload a log file to logs bucket."""
        key = s3_key or local_path.name
        return self._put(BUCKET_LOGS, local_path, key)

    def put_db(self) -> bool:
        """Sync the DuckDB database to cell-data."""
        db_path = Path("/home/placer/sierra_prospector/db/prospector.duckdb")
        return self._put(BUCKET_CELLDATA, db_path, "db/prospector.duckdb")

    def _put(self, bucket: str, local_path: Path, s3_key: str) -> bool:
        if not self._enabled:
            log.warning("Storage not enabled — upload skipped", key=s3_key)
            return False
        try:
            log.info("Uploading", bucket=bucket, key=s3_key)
            self._client.upload_file(str(local_path), bucket, s3_key)
            log.info("Upload complete", key=s3_key)
            return True
        except Exception as e:
            log.error("Upload failed", bucket=bucket, key=s3_key, error=str(e))
            return False

    # ── List ──────────────────────────────────────────────────────────────────

    def list_geodata(self, prefix: str = "") -> List[str]:
        return self._list(BUCKET_GEODATA, prefix)

    def list_celldata(self, prefix: str = "") -> List[str]:
        return self._list(BUCKET_CELLDATA, prefix)

    def _list(self, bucket: str, prefix: str) -> List[str]:
        if not self._enabled:
            return []
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
            return keys
        except Exception as e:
            log.error("List failed", bucket=bucket, error=str(e))
            return []

    # ── Cache ─────────────────────────────────────────────────────────────────

    def cache_size_bytes(self) -> int:
        return sum(f.stat().st_size for f in CACHE_DIR.rglob("*") if f.is_file())

    def evict_lru(self, target_bytes: int = MAX_CACHE_BYTES):
        current = self.cache_size_bytes()
        if current <= target_bytes:
            return
        files = sorted(
            [f for f in CACHE_DIR.rglob("*") if f.is_file()
             and "db/" not in str(f)],
            key=lambda f: f.stat().st_atime
        )
        freed = 0
        for f in files:
            if current - freed <= target_bytes:
                break
            freed += f.stat().st_size
            f.unlink()
        log.info("Cache eviction complete", freed_mb=round(freed/1024/1024, 1))


# Module singleton
storage = StorageManager()
