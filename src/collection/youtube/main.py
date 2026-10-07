#!/usr/bin/env python3
"""
YouTube Shorts dataset builder.

Usage examples
--------------
# Collect 10,000 cooking Shorts (default settings):
    python main.py --topic cooking

# Multiple topics with extra keywords:
    python main.py --topic cooking baking --extra-keywords easy beginner

# Use a JSON config file + override target:
    python main.py --config config.json --target 5000

# Resume an interrupted session (just re-run the same command):
    python main.py --topic cooking

# Only export raw.jsonl → dataset.jsonl without scraping (dedup only):
    python main.py --export-only

# Print current progress:
    python main.py --stats
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time

from tqdm import tqdm

from .config import ScraperConfig, load_config
from .scraper import YtDlpScraper, generate_queries
from .storage import DataStore
from .transcriber import TranscriptManager

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect YouTube Shorts metadata into a local dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--topic", nargs="+", dest="topics", metavar="TOPIC",
        help="Hashtag topic(s) to search (without #). E.g. cooking baking fitness",
    )
    p.add_argument(
        "--extra-keywords", nargs="+", default=[], metavar="KW",
        help="Extra keywords mixed into queries for more variety.",
    )
    p.add_argument(
        "--target", type=int, metavar="N",
        help="Target number of unique records (default: 10000).",
    )
    p.add_argument(
        "--results-per-query", type=int, metavar="N",
        help="Max results per yt-dlp search call (default: 500).",
    )
    p.add_argument(
        "--sleep", type=float, dest="sleep_between_queries", metavar="S",
        help="Seconds to sleep between search queries (default: 4.0).",
    )
    p.add_argument(
        "--data-dir", type=str, metavar="DIR",
        help="Directory for output files (default: data/).",
    )
    p.add_argument(
        "--config", type=str, metavar="FILE",
        help="Path to a JSON config file.",
    )
    p.add_argument(
        "--export-only", action="store_true",
        help="Skip scraping; only export raw.jsonl → dataset.jsonl.",
    )
    p.add_argument(
        "--stats", action="store_true",
        help="Print collection stats and exit.",
    )
    p.add_argument(
        "--whisper", action="store_true",
        help="Enable WhisperX transcription with speaker diarization.",
    )
    p.add_argument(
        "--whisper-model", type=str, metavar="MODEL",
        help="WhisperX model size: tiny|base|small|medium|large-v2 (default: base).",
    )
    p.add_argument(
        "--whisper-device", type=str, metavar="DEVICE",
        help="Device for WhisperX: cpu|cuda|mps (default: cpu).",
    )
    p.add_argument(
        "--hf-token", type=str, metavar="TOKEN",
        help="HuggingFace token for WhisperX speaker diarization.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Show yt-dlp output and debug logging.",
    )
    return p


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def print_stats(store: DataStore, config: ScraperConfig) -> None:
    count = store.count()
    pct = count / config.target_count * 100 if config.target_count else 0
    print(f"Collected : {count:>7,}  /  {config.target_count:,}  ({pct:.1f}%)")
    print(f"Data dir  : {config.data_dir}/")


def run(config: ScraperConfig) -> None:
    store = DataStore(config)
    scraper = YtDlpScraper(config)
    transcripts = TranscriptManager(config)

    if config.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s [%(levelname)s] %(message)s",
        )

    already = store.count()
    if already >= config.target_count:
        print(f"Target already reached: {already:,} records collected.")
        store.export()
        return

    queries = generate_queries(config.topics, config.extra_keywords)
    # Shuffle so a resumed run doesn't always start from the same query.
    random.shuffle(queries)

    print(
        f"Starting collection │ target={config.target_count:,} │ "
        f"already={already:,} │ queries={len(queries)}"
    )

    with tqdm(
        total=config.target_count,
        initial=already,
        desc="Shorts collected",
        unit="video",
        dynamic_ncols=True,
    ) as pbar:

        for query, topic in queries:
            if store.count() >= config.target_count:
                break

            pbar.set_postfix({"q": query[:35], "total": store.count()})

            entries = scraper.fetch_with_backoff(query)
            new_this_batch = 0

            for entry in entries:
                if store.count() >= config.target_count:
                    break
                # Phase 1: fast filters on flat metadata (no extra HTTP requests)
                video_id = entry.get("id") or entry.get("videoId")
                if not video_id or store.is_seen(video_id):
                    continue
                if not scraper.is_duration_candidate(entry):
                    continue
                if not scraper.is_viral_candidate(entry):
                    continue
                if not scraper.is_language_valid(entry):
                    continue
                    
                # Phase 2: HEAD request to confirm it's a Short (200 = Short, 303 = not)
                if not scraper.is_short(video_id):
                    continue
                # Phase 3: full extraction to get rich metadata
                full_entry = scraper.fetch_full_info(video_id)
                if not full_entry:
                    continue
                record = scraper.parse_record(full_entry, query, topic)
                if record and store.save_record(record):
                    pbar.update(1)
                    new_this_batch += 1
                    transcripts.save_yt_transcript(video_id)
                    if config.use_whisper:
                        transcripts.save_whisper_transcript(video_id)

            # Adaptive sleep: very few new results → likely rate-limited or
            # query exhausted, sleep longer before the next request.
            if new_this_batch < 10:
                sleep_time = config.sleep_between_queries * 3
            else:
                sleep_time = config.sleep_between_queries

            if config.verbose:
                tqdm.write(
                    f"  query={query!r}  new={new_this_batch}  "
                    f"total={store.count()}  sleep={sleep_time:.1f}s"
                )

            time.sleep(sleep_time)

    final = store.count()
    print(f"\nCollection done: {final:,} unique Shorts saved.")
    store.export()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Build CLI overrides dict (None values are ignored by load_config)
    overrides = {
        "topics": args.topics,
        "extra_keywords": args.extra_keywords or None,
        "target_count": args.target,
        "results_per_query": args.results_per_query,
        "sleep_between_queries": args.sleep_between_queries,
        "data_dir": args.data_dir,
        "verbose": True if args.verbose else None,
        "use_whisper": True if args.whisper else None,
        "whisper_model": args.whisper_model,
        "whisper_device": args.whisper_device,
        "whisper_hf_token": args.hf_token,
    }
    # Remove None values so load_config defaults are preserved
    overrides = {k: v for k, v in overrides.items() if v is not None}

    config = load_config(path=args.config, overrides=overrides)

    if args.stats:
        store = DataStore(config)
        print_stats(store, config)
        sys.exit(0)

    if args.export_only:
        store = DataStore(config)
        store.export()
        sys.exit(0)

    run(config)


if __name__ == "__main__":
    main()
