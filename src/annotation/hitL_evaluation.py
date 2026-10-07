import pandas as pd
import pymongo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configurazione MongoDB
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"

DATA_LAKE_DIR = Path("data_lake")
HUMAN_COMPLETED_FILE = DATA_LAKE_DIR / "human_review_COMPLETED.csv"

def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def evaluate_hitl():
    if not HUMAN_COMPLETED_FILE.exists():
        print("❌ File completato non trovato. Assicurati di aver compilato e salvato human_review_COMPLETED.csv")
        return

    # 1. Carichiamo le etichette manuali umane
    human_df = pd.read_csv(HUMAN_COMPLETED_FILE)
    human_df['human_label'] = human_df['human_label'].str.strip().str.lower()
    
    # Rimuoviamo eventuali righe lasciate vuote dall'umano
    human_df = human_df[human_df['human_label'] != ""]
    
    if len(human_df) == 0:
         print("❌ Nessuna etichetta umana valida trovata nel file.")
         return

    # 2. Recuperiamo le etichette LLM corrispondenti direttamente da MongoDB
    video_ids = human_df['video_id'].astype(str).tolist()
    
    collection = get_mongo_collection()
    llm_records = list(collection.find(
        {"video_id": {"$in": video_ids}},
        {"_id": 0, "video_id": 1, "llm_misinfo_consensus": 1}
    ))
    
    llm_df = pd.DataFrame(llm_records)
    llm_df['video_id'] = llm_df['video_id'].astype(str)
    
    # 3. Uniamo i dataset
    merged_df = pd.merge(human_df, llm_df, on='video_id', how='inner')
    
    if len(merged_df) == 0:
        print("❌ Impossibile unire i dataset. Controlla che i video_id nel file CSV esistano in MongoDB.")
        return
    
    y_true = merged_df['human_label'] # Il Ground Truth Umano
    y_pred = merged_df['llm_misinfo_consensus'] # La predizione ufficiale in MongoDB
    
    # --- REPORT STATISTICO ---
    accuracy = accuracy_score(y_true, y_pred)
    print("="*50)
    print(f"🎯 ACCURACY DEL SISTEMA LLM (MongoDB Base): {accuracy * 100:.1f}%")
    print("="*50)
    
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Matrice di Confusione
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Matrice di Confusione: Umano vs LLM Consensus')
    plt.ylabel('Ground Truth (Umano)')
    plt.xlabel('Predizione (LLM - MongoDB)')
    
    output_img = DATA_LAKE_DIR / "confusion_matrix.png"
    plt.savefig(output_img)
    print(f"\n🖼️ Matrice di confusione salvata in: {output_img}")

if __name__ == "__main__":
    evaluate_hitl()