import streamlit as st
import pandas as pd
from utils import get_data

# Configurazione base della pagina (DEVE essere la prima chiamata Streamlit)
st.set_page_config(
    page_title="Short Video Misinformation",
    page_icon="📱",
    layout="wide"
)

def main():
    st.title("Short-form Video Analytics Dashboard")
    st.markdown("""
    Benvenuto nella dashboard analitica del progetto di Big Data Engineering. 
    Questa piattaforma permette di esplorare e analizzare i dati estratti dalle principali piattaforme di video brevi (YouTube Shorts, TikTok, Instagram Reels) per studiare i trend tematici e l'engagement.
    """)

    # 1. Caricamento Dati
    with st.spinner("Connessione al Data Lake (MongoDB) in corso..."):
        df = get_data()

    if df.empty:
        st.error("Nessun dato trovato nel database. Esegui prima la pipeline ETL!")
        return

    # 2. Metriche Generali (Overview)
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Video Totali", f"{len(df):,}")
    col2.metric("Piattaforme", df['platform'].nunique() if 'platform' in df.columns else 0)
    col3.metric("Topic Analizzati", df['topic'].nunique() if 'topic' in df.columns else 0)
    
    total_views = df['view_count'].sum() if 'view_count' in df.columns else 0
    # Formattazione per rendere leggibili i milioni/miliardi
    if total_views > 1e9:
        views_str = f"{total_views/1e9:.1f}B"
    elif total_views > 1e6:
        views_str = f"{total_views/1e6:.1f}M"
    else:
        views_str = f"{total_views:,}"
    col4.metric("Visualizzazioni Totali", views_str)

    st.divider()

    # 3. Esplorazione Raw Data & Motore di Ricerca (Requisito della traccia)
    st.header("🔍 Esplorazione e Ricerca Dataset")
    st.markdown("Usa i filtri o la barra di ricerca per ispezionare i metadati grezzi e le trascrizioni ASR dei video.")
    
    # Barra di ricerca testuale
    search_query = st.text_input("Cerca per parola chiave nel titolo o nella trascrizione:", "")
    
    # Filtri di base
    f_col1, f_col2 = st.columns(2)
    platforms = df['platform'].unique() if 'platform' in df.columns else []
    topics = df['topic'].unique() if 'topic' in df.columns else []
    
    selected_platform = f_col1.multiselect("Filtra per Piattaforma", platforms, default=platforms)
    selected_topic = f_col2.multiselect("Filtra per Topic", topics, default=topics)

    # Applicazione filtri
    filtered_df = df.copy()
    if selected_platform:
        filtered_df = filtered_df[filtered_df['platform'].isin(selected_platform)]
    if selected_topic:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topic)]
        
    if search_query:
        # Cerca nel titolo o nel testo trascritto (case-insensitive)
        mask = filtered_df['title'].str.contains(search_query, case=False, na=False) | \
               filtered_df['transcript'].str.contains(search_query, case=False, na=False)
        filtered_df = filtered_df[mask]

    # Mostriamo la tabella (limitiamo a 100 righe per performance)
    st.dataframe(
        filtered_df[['video_id', 'platform', 'topic', 'title', 'view_count', 'upload_date']].head(100),
        use_container_width=True
    )

    # Dettaglio Trascrizione
    st.subheader("Ispeziona Trascrizione")
    
    # 1. Pulsante di svuotamento cache nella Sidebar
    if st.sidebar.button("🔄 Aggiorna Dati Database"):
        st.cache_data.clear()
        st.rerun()

    # 2. Creiamo un dizionario { "Titolo Video (ID)": "video_id" } per la selectbox
    # Usiamo il titolo per comodità visiva, troncandolo a 60 caratteri se troppo lungo
    video_dict = {}
    for _, row in filtered_df.iterrows():
        title = str(row.get('title', 'N/A'))
        short_title = title[:60] + "..." if len(title) > 60 else title
        video_dict[f"{short_title} ({row['video_id']})"] = row['video_id']

    selected_display = st.selectbox("Seleziona un video per leggerne il testo:", options=list(video_dict.keys()))
    
    if selected_display:
        selected_video_id = video_dict[selected_display]
        video_data = filtered_df[filtered_df['video_id'] == selected_video_id].iloc[0]
        st.info(f"**Titolo:** {video_data.get('title', 'N/A')}")
        st.write(video_data.get('transcript', 'Nessuna trascrizione disponibile.'))

if __name__ == "__main__":
    main()