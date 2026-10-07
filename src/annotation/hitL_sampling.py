import pandas as pd
import pymongo
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

DATA_LAKE_DIR = Path("data_lake")
HUMAN_TASK_FILE = DATA_LAKE_DIR / "human_review.csv"

# Credenziali MongoDB
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"

def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def generate_stratified_sample(sample_size=20):
    logging.info("🎯 Avvio estrazione campione stratificato (da MongoDB) per HITL...")
    
    collection = get_mongo_collection()
    
    # 1. Recupera solo i video che hanno già un consenso LLM valido
    valid_labels = ["reliable", "misleading", "false", "satire", "out-of-context"]
    
    raw_data = list(collection.find(
        {"llm_misinfo_consensus": {"$in": valid_labels}},
        {"_id": 0, "video_id": 1, "title": 1, "platform": 1, "topic": 1, "url": 1, "llm_misinfo_consensus": 1}
    ))
    
    if not raw_data:
        logging.error("Nessun video annotato trovato in MongoDB per il campionamento.")
        return
        
    df_valid = pd.DataFrame(raw_data)

    # 2. Campionamento Stratificato
    actual_sample_size = min(sample_size, len(df_valid))
    fraction = actual_sample_size / len(df_valid)
    
    # Campionamento bilanciato basato sul consenso LLM
    sample_df = df_valid.groupby('llm_misinfo_consensus', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), max(1, int(round(len(x) * fraction)))), random_state=42)
    )
    
    # Correzione arrotondamenti
    if len(sample_df) > actual_sample_size:
        sample_df = sample_df.sample(n=actual_sample_size, random_state=42)
        
    # 3. Preparazione file per revisione umana (Nascondiamo il consenso LLM!)
    human_df = sample_df[['video_id', 'title', 'platform', 'topic', 'url']].copy()
    human_df['human_label'] = ""
    human_df['human_notes'] = ""
    
    human_df.to_csv(HUMAN_TASK_FILE, index=False)
    
    logging.info(f"✅ Campione di {len(sample_df)} video estratto con successo.")
    logging.info(f"📝 File salvato in: {HUMAN_TASK_FILE}")
    logging.info("\nProporzioni LLM nel campione nascosto:")
    print(df_valid.loc[sample_df.index, 'llm_misinfo_consensus'].value_counts())
if __name__ == "__main__":
    generate_stratified_sample(sample_size=50)