import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import krippendorff
import logging
import pymongo

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Metrics")

# --- CONFIGURAZIONE MONGODB ---
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"

DATA_LAKE_DIR = Path("data_lake")
ANNOTATIONS_DIR = DATA_LAKE_DIR / "annotations"
OUTPUT_FILE = DATA_LAKE_DIR / "final_consensus_dataset.csv"

ALLOWED_LABELS = ["reliable", "misleading", "false", "satire", "out-of-context"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(ALLOWED_LABELS)}
LABEL_TO_ID["error"] = np.nan 

def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def main():
    logger.info("📊 Avvio calcolo delle metriche di Accordo (IAA)...")
    
    # 1. RECUPERO DEL DATASET UFFICIALE DA MONGODB
    collection = get_mongo_collection()
    
    # Estraiamo tutti i video che hanno ricevuto un'annotazione di consenso
    db_records = list(collection.find(
        {"llm_misinfo_consensus": {"$exists": True}},
        {"_id": 0, "video_id": 1, "llm_misinfo_consensus": 1}
    ))
    
    if not db_records:
        logger.error("Nessun video annotato (llm_misinfo_consensus) trovato in MongoDB.")
        return
        
    logger.info(f"✅ Trovati {len(db_records)} record consolidati in MongoDB.")

    # 2. CALCOLO DELLE METRICHE DAI LOG JSON LOCALI
    json_files = list(ANNOTATIONS_DIR.glob("*_annotation.json"))
    judges = []
    
    # Trovare dinamicamente i nomi dei giudici
    for f in json_files:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            if "judgments" in data:
                judges = [j for j in data["judgments"].keys() if not j.startswith("TieBreak_")]
                break
                
    if not judges:
        logger.warning("Nessun log di giudici trovato. Salto il calcolo dell'Alpha di Krippendorff.")
    else:
        krippendorff_matrix = np.full((len(judges), len(json_files)), np.nan)
        
        # Inizializziamo un dizionario per memorizzare i voti disaggregati da unire al CSV
        votes_dict = {doc["video_id"]: {} for doc in db_records}

        for col_idx, file_path in enumerate(json_files):
            with open(file_path, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
                
            vid = video_data.get("video_id")
            judgments = video_data.get("judgments", {})
            
            for row_idx, judge_name in enumerate(judges):
                judge_label = judgments.get(judge_name, {}).get("label", "error")
                krippendorff_matrix[row_idx, col_idx] = LABEL_TO_ID.get(judge_label, np.nan)
                
                # Salviamo il voto per incollarlo nel DataFrame finale
                if vid in votes_dict:
                    votes_dict[vid][f"vote_{judge_name}"] = judge_label

    # 3. CREAZIONE DEL DATASET FINALE (Merge DB + Voti)
    final_rows = []
    for record in db_records:
        row = {
            "video_id": record["video_id"],
            "consensus_label": record["llm_misinfo_consensus"] # LA VERITÀ DA MONGODB
        }
        # Aggiungiamo i voti dei singoli giudici se li abbiamo estratti dai JSON
        if record["video_id"] in votes_dict:
            row.update(votes_dict[record["video_id"]])
            
        final_rows.append(row)

    df = pd.DataFrame(final_rows)
    df.to_csv(OUTPUT_FILE, index=False)
    
    # --- REPORT STATISTICO ---
    logger.info("\n" + "="*50)
    logger.info("🏆 REPORT FINALE LLM-as-a-Judge")
    logger.info("="*50)
    
    metrics_export = {
        "cohen_kappa": {},
        "krippendorff_alpha": None,
        "alpha_evaluation": ""
    }
        
    if judges:
        logger.info("\n🤝 Accordo a Coppie (Cohen's Kappa):")
        for i in range(len(judges)):
            for j in range(i + 1, len(judges)):
                judge1, judge2 = judges[i], judges[j]
                
                # Sicurezza: controlliamo se le colonne dei voti esistono nel DataFrame
                col1 = f"vote_{judge1}"
                col2 = f"vote_{judge2}"
                
                if col1 in df.columns and col2 in df.columns:
                    valid_mask = (df[col1].isin(ALLOWED_LABELS)) & (df[col2].isin(ALLOWED_LABELS))
                    y1 = df.loc[valid_mask, col1]
                    y2 = df.loc[valid_mask, col2]
                    
                    if len(y1) > 0:
                        kappa = cohen_kappa_score(y1, y2)
                        pair_name = f"{judge1.replace('Judge_', '')} vs {judge2.replace('Judge_', '')}"
                        metrics_export["cohen_kappa"][pair_name] = kappa
                        logger.info(f" - {pair_name}: {kappa:.3f}")

        logger.info("\n👑 Accordo Globale (Krippendorff's Alpha):")
        try:
            alpha = krippendorff.alpha(reliability_data=krippendorff_matrix, level_of_measurement='nominal')
            metrics_export["krippendorff_alpha"] = alpha
            logger.info(f" - Alpha = {alpha:.3f}")
            
            if alpha >= 0.800:
                metrics_export["alpha_evaluation"] = "Eccellente (Affidabilità altissima)"
            elif alpha >= 0.500:
                metrics_export["alpha_evaluation"] = "Discreta (Conclusioni provvisorie)"
            else:
                metrics_export["alpha_evaluation"] = "Bassa (Alto disaccordo)"
                
            logger.info(f"   Valutazione: {metrics_export['alpha_evaluation']}")
        except Exception as e:
            logger.error(f"   Impossibile calcolare Alpha: {e}")

        METRICS_JSON_FILE = DATA_LAKE_DIR / "agreement_metrics.json"
        with open(METRICS_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics_export, f, indent=4)

    logger.info(f"\n💾 Dataset CSV finale creato (Sorgente: MongoDB): {OUTPUT_FILE}")

if __name__ == "__main__":
    main()