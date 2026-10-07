import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional

@dataclass
class ScraperConfig:
    topics: List[str] = field(default_factory=lambda: ["finance"])
    extra_keywords: List[str] = field(default_factory=list)
    target_count: int = 10
    results_per_query: int = 100
    max_duration: int = 90  # I Reels possono arrivare a 90 secondi

    # --- UNIFICATO CON IL DATA LAKE GLOBALE ---
    data_dir: str = "data_lake"
    jsonl_filename: str = "raw.jsonl"
    seen_ids_filename: str = "seen_ids.txt"
    output_jsonl: str = "dataset.jsonl"

    use_whisper: bool = False
    whisper_model: str = "base"
    whisper_device: str = "cpu"

def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> ScraperConfig:
    """Fonde le configurazioni di default con quelle passate da terminale."""
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