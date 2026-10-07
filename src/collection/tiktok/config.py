from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ScraperConfig:
    # --- Targeting ---
    topics: List[str] = field(default_factory=lambda: ["cooking"])
    extra_keywords: List[str] = field(default_factory=list)

    # --- Volume ---
    target_count: int = 10_000
    results_per_query: int = 500
    max_duration: int = 60  # seconds; same cap as YouTube Shorts

    # --- Rate limiting ---
    sleep_between_queries: float = 6.0   # TikTok is more aggressive than YouTube
    backoff_base: float = 60.0           # first retry wait in seconds
    backoff_max_wait: float = 600.0      # cap on retry wait
    max_retries: int = 5

    # --- Storage ---
    data_dir: str = "data_lake"
    jsonl_filename: str = "raw.jsonl"
    seen_ids_filename: str = "seen_ids.txt"
    output_jsonl: str = "dataset.jsonl"

    # --- Optional cookies (improve stability, no account required) ---
    # Copy from browser DevTools → Application → Cookies → tiktok.com
    ms_token: Optional[str] = None    # msToken cookie (anonymous visitor, auto-set by TikTok)
    session_id: Optional[str] = None  # sessionid cookie (requires a free account)

    # --- Comments (on by default; all pages are fetched via cursor pagination) ---
    fetch_comments: bool = True

    # --- WhisperX (speaker diarization, opt-in) ---
    use_whisper: bool = False
    whisper_model: str = "base"           # tiny | base | small | medium | large-v2
    whisper_device: str = "cpu"           # cpu | cuda
    whisper_language: Optional[str] = None  # None = auto-detect
    whisper_hf_token: str = ""            # HuggingFace token (required for diarization)

    # --- Behaviour ---
    verbose: bool = False


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> ScraperConfig:
    """
    Build a ScraperConfig by optionally loading a JSON file and applying
    a dict of overrides (e.g. from CLI args).

    Priority: overrides > JSON file > dataclass defaults.
    """
    base: dict = asdict(ScraperConfig())

    if path:
        with open(path, "r", encoding="utf-8") as f:
            file_data = json.load(f)
        for k, v in file_data.items():
            if k in base:
                base[k] = v

    if overrides:
        for k, v in overrides.items():
            if v is not None and k in base:
                base[k] = v

    return ScraperConfig(**base)
