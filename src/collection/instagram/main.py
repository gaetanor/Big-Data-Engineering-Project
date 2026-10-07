import argparse
from tqdm import tqdm
from .config import ScraperConfig, load_config
from .storage import DataStore
from .scraper import InstagramScraper, generate_queries
from .transcriber import TranscriptManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", nargs="+", required=True)
    parser.add_argument("--extra-keywords", nargs="+", default=[])
    parser.add_argument("--target", type=int, default=10)
    parser.add_argument("--results-per-query", type=int, default=100)
    parser.add_argument("--data-dir", type=str)
    args = parser.parse_args()

    # Usiamo il dizionario overrides per passare solo i valori effettivamente inseriti
    overrides = {
        "topics": args.topic,
        "extra_keywords": args.extra_keywords or None,
        "target_count": args.target,
        "results_per_query": args.results_per_query,
        "data_dir": args.data_dir
    }
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # Carichiamo la configurazione usando la nuova funzione
    config = load_config(overrides=overrides)
    
    store = DataStore(config)
    transcriber = TranscriptManager(config)
    scraper = InstagramScraper(config)

    already = store.count()
    if already >= config.target_count:
        print(f"Target già raggiunto: {already}")
        return

    queries = generate_queries(config.topics, config.extra_keywords)
    print(f"Avvio Instagram Scraper | Target: {config.target_count}")

    with tqdm(total=config.target_count, initial=already) as pbar:
        for hashtag, topic in queries:
            if store.count() >= config.target_count: break
            
            urls = scraper.fetch_reels_urls(hashtag, 50)
            tqdm.write(f"\n[DEBUG] Trovati {len(urls)} link per l'hashtag #{hashtag}")
            
            for url in urls:
                if store.count() >= config.target_count: break
                
                info = scraper.fetch_full_info(url)
                if not info: continue
                
                vid = info.get("id")
                if not vid or store.is_seen(vid): continue
                
                if info.get("duration", 0) > config.max_duration: 
                    tqdm.write(f"  [Scartato] {vid}: Troppo lungo ({info.get('duration')}s)")
                    continue
                
                tqdm.write(f"  [OK] Download Reel {vid} in corso...")
                tqdm.write(f"  [*] Estrazione commenti e visualizzazioni in corso...")
                
                # CHIAMATA AGGIORNATA: Estrae sia i commenti che le views!
                extra_data = scraper.get_extra_data(url)
                
                # Passiamo l'oggetto extra_data alla funzione
                record = scraper.parse_record(info, topic, extra_data)
                
                if store.save_record(record):
                    pbar.update(1) # Ora la barra si aggiornerà fluidamente!
                    transcriber.save_transcript(vid, url)

    scraper.close()
    print("Raccolta Instagram completata!")

if __name__ == "__main__":
    main()