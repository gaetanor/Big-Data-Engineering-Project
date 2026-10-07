import streamlit as st
import pymongo
import pandas as pd

# Connessione al DB locale (come impostato nel docker-compose)
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"

@st.cache_resource
def init_connection():
    """
    Inizializza la connessione a MongoDB. 
    @st.cache_resource garantisce che il client venga creato UNA SOLA VOLTA
    per tutta la durata dell'applicazione, ottimizzando drasticamente le performance.
    """
    return pymongo.MongoClient(MONGO_URI)

@st.cache_data(ttl=3600)
def get_data():
    """
    Estrae i dati da MongoDB e li converte in un DataFrame Pandas.
    @st.cache_data(ttl=3600) memorizza in cache i dati estratti per 1 ora.
    Se l'utente cambia pagina o applica un filtro, non interroghiamo di nuovo il DB.
    """
    client = init_connection()
    db = client["short_videos_db"]
    collection = db["videos"]
    
    # Estraiamo tutti i documenti (escludendo l'ID di Mongo per pulizia)
    data = list(collection.find({}, {"_id": 0}))
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    
    # Convertiamo le date in formato datetime di Pandas per le analisi temporali
    if 'upload_date' in df.columns:
        # Molte API restituiscono date in formati diversi (es. YYYYMMDD per yt-dlp)
        df['upload_date'] = pd.to_datetime(df['upload_date'], format='mixed', errors='coerce')
        
    # Puliamo l'engagement forzandolo a numerico (sostituendo i NaN con 0)
    for col in ['view_count', 'like_count', 'comment_count', 'share_count']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    return df