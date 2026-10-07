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
    target_count: int = 10
    results_per_query: int = 200
    max_duration: int = 60  # seconds; Shorts are ≤ 60s

    # --- Rate limiting ---
    sleep_between_queries: float = 4.0   # seconds between search batches
    backoff_base: float = 30.0           # first retry wait in seconds
    backoff_max_wait: float = 300.0      # cap on retry wait
    max_retries: int = 5

    # --- Storage ---
    data_dir: str = "data_lake"
    jsonl_filename: str = "raw.jsonl"
    seen_ids_filename: str = "seen_ids.txt"
    output_jsonl: str = "dataset.jsonl"

    # --- Transcripts (yt-dlp auto-captions) ---
    transcript_langs: List[str] = field(default_factory=lambda: ["it", "en"])

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
        # Only keep known keys so stray JSON fields don't crash the dataclass
        for k, v in file_data.items():
            if k in base:
                base[k] = v

    if overrides:
        for k, v in overrides.items():
            if v is not None and k in base:
                base[k] = v

    return ScraperConfig(**base)
