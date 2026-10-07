import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, coalesce, lit
import datetime
import pymongo
from pymongo import ReplaceOne

# --- CONFIGURAZIONE PATH ---
DATA_LAKE_DIR = Path("data_lake")
METADATA_DIR = str(DATA_LAKE_DIR / "raw_metadata" / "*.json")
TRANSCRIPTS_DIR = str(DATA_LAKE_DIR / "transcripts" / "*.txt")

# --- CONFIGURAZIONE MONGODB ---
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"

def write_to_mongo(partition):
    """
    Funzione eseguita sui worker di Spark per scrivere i dati direttamente
    in MongoDB usando pymongo con operazioni BULK ad alta efficienza.
    """
    # Connessione locale per ogni partizione
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    bulk_operations = [] # Lista per accumulare le operazioni
    
    for row in partition:
        # Convertiamo la Row di Spark in un dizionario Python standard
        doc = row.asDict(recursive=True)
        
        # Se manca il video_id, scartiamo il record per non corrompere MongoDB
        if not doc.get("video_id"):
            continue
            
        # Usiamo video_id come chiave primaria (_id)
        doc["_id"] = doc["video_id"]
        
        # Invece di eseguire subito, prepariamo un'operazione ReplaceOne
        bulk_operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=True))
        
        # Eseguiamo a blocchi di 1000 per non saturare la memoria del worker
        if len(bulk_operations) >= 1000:
            try:
                collection.bulk_write(bulk_operations, ordered=False)
            except Exception as e:
                print(f"Errore durante il bulk write parziale: {e}")
            finally:
                bulk_operations.clear() # Svuotiamo la lista per il blocco successivo
                
    # Eseguiamo le operazioni rimanenti
    if bulk_operations:
        try:
            collection.bulk_write(bulk_operations, ordered=False)
        except Exception as e:
            print(f"Errore durante il bulk write finale: {e}")
            
    client.close()

def run_etl_pipeline():
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Avvio PySpark ETL Pipeline (PyMongo Mode)...")

    # 1. Inizializzazione Spark Session pulita
    spark = SparkSession.builder \
        .appName("ShortVideoMisinformation_ETL") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    try:
        # 2. EXTRACT: Lettura dei Metadati JSON
        print(" -> Estrazione Metadati JSON...")
        # Aggiungiamo dropDuplicates per sicurezza
        df_metadata = spark.read.json(METADATA_DIR, multiLine=True).dropDuplicates(["video_id"])
        
        # 3. EXTRACT: Lettura delle Trascrizioni TXT
        print(" -> Estrazione Trascrizioni Audio...")
        rdd_transcripts = spark.sparkContext.wholeTextFiles(TRANSCRIPTS_DIR)
        
        if rdd_transcripts.isEmpty():
            print("    [!] Nessuna trascrizione trovata. Procedo solo con i metadati.")
            df_transcripts = spark.createDataFrame([], schema="path STRING, transcript STRING")
        else:
            df_transcripts = rdd_transcripts.toDF(["path", "transcript"])

        df_transcripts = df_transcripts.withColumn("video_id", regexp_extract(col("path"), r"([^/]+)\.txt$", 1))
        df_transcripts = df_transcripts.drop("path").dropDuplicates(["video_id"])

        # 4. TRANSFORM: Join e Data Cleaning
        print(" -> Trasformazione e Unificazione Schema (Join)...")
        df_unified = df_metadata.join(df_transcripts, on="video_id", how="left")
        
        # Gestione dei valori nulli
        if "transcript" in df_unified.columns:
            df_unified = df_unified.withColumn("transcript", coalesce(col("transcript"), lit("")))
        else:
            df_unified = df_unified.withColumn("transcript", lit(""))
            
        if "description" in df_unified.columns:
            df_unified = df_unified.withColumn("description", coalesce(col("description"), lit("")))
        else:
            df_unified = df_unified.withColumn("description", lit(""))

        record_count = df_unified.count()
        print(f" -> Trovati {record_count} record validi pronti per il caricamento.")

        # 5. LOAD: Scrittura su MongoDB via PyMongo
        if record_count > 0:
            print(" -> Ottimizzazione partizioni e Caricamento su MongoDB in Bulk...")
            
            # Riduciamo a 4 partizioni per aprire solo 4 connessioni concorrenti a MongoDB
            df_unified_optimized = df_unified.coalesce(4)
            
            df_unified_optimized.foreachPartition(write_to_mongo)
            
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Pipeline ETL completata con successo! Dati disponibili in MongoDB.")
        else:
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Nessun dato da caricare. Controlla la cartella raw_metadata.")
            
    except Exception as e:
        print(f"\n[ERRORE FATALE] La pipeline si è interrotta: {e}")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    run_etl_pipeline()