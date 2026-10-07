from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, Set

from .config import ScraperConfig


class DataStore:
    """
    Crash-safe, resumable storage for TikTok video metadata.

    Layout inside data_dir:
      raw.jsonl      – one JSON object per line, append-only
      seen_ids.txt   – one video_id per line, used as a fast dedup index
      dataset.jsonl  – final export (written on demand)
    """

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.data_dir = pathlib.Path(config.data_dir)
        self.jsonl_path = self.data_dir / config.jsonl_filename
        self.seen_ids_path = self.data_dir / config.seen_ids_filename
        self.raw_metadata_dir = self.data_dir / "raw_metadata"
        self._seen_ids: Set[str] = set()
        self._ensure_dirs()
        self._load_seen_ids()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_metadata_dir.mkdir(parents=True, exist_ok=True)

    def _load_seen_ids(self) -> None:
        """
        Load seen_ids.txt into memory.
        If the file is missing (e.g. after a crash), reconstruct it from
        raw.jsonl so we never re-collect a video we already have.
        """
        if self.seen_ids_path.exists():
            with open(self.seen_ids_path, "r", encoding="utf-8") as f:
                for line in f:
                    vid = line.strip()
                    if vid:
                        self._seen_ids.add(vid)
        elif self.jsonl_path.exists():
            # Crash recovery: rebuild seen_ids.txt from raw.jsonl
            reconstructed: Set[str] = set()
            with open(self.jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        vid = record.get("video_id")
                        if vid:
                            reconstructed.add(vid)
                    except json.JSONDecodeError:
                        pass
            self._seen_ids = reconstructed
            with open(self.seen_ids_path, "w", encoding="utf-8") as f:
                for vid in self._seen_ids:
                    f.write(vid + "\n")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_seen(self, video_id: str) -> bool:
        return video_id in self._seen_ids

    def save_record(self, record: Dict[str, Any]) -> bool:
        """
        Persist a record.  Returns True if saved, False if duplicate.

        Write order: JSONL first, then register ID.
        A crash between the two steps leaves a duplicate in raw.jsonl
        which export() deduplicates via seen-set logic.
        """
        vid = record.get("video_id")
        if not vid or vid in self._seen_ids:
            return False
        json_path = self.raw_metadata_dir / f"tiktok_{vid}.json"
        import json
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=4)
            
        self._append_jsonl(record)
        self._register_seen(vid)
        return True

    def count(self) -> int:
        return len(self._seen_ids)

    def export(self) -> None:
        """Read raw.jsonl → deduplicate → write dataset.jsonl."""
        if not self.jsonl_path.exists():
            print("Nothing to export: raw.jsonl not found.")
            return

        records = []
        seen: Set[str] = set()
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    vid = record.get("video_id")
                    if vid and vid not in seen:
                        seen.add(vid)
                        records.append(record)
                except json.JSONDecodeError:
                    pass

        if not records:
            print("Nothing to export: raw.jsonl is empty.")
            return

        out = self.data_dir / self.config.output_jsonl
        with open(out, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"JSONL written → {out}  ({len(records):,} rows)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _append_jsonl(self, record: Dict[str, Any]) -> None:
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _register_seen(self, video_id: str) -> None:
        self._seen_ids.add(video_id)
        with open(self.seen_ids_path, "a", encoding="utf-8") as f:
            f.write(video_id + "\n")
