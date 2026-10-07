import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Importiamo la funzione di connessione
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data

st.set_page_config(page_title="Cross-Platform Overlap", page_icon="🔗", layout="wide")

def normalize_text(text):
    """Pulisce il testo per l'algoritmo di similitudine."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Rimuove hashtag comuni che "sporcano" la similitudine
    text = re.sub(r'#(shorts|reels|tiktok|viral|fyp)\b', '', text)
    # Rimuove punteggiatura e caratteri speciali
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

@st.cache_data
def detect_near_duplicates(df, similarity_threshold=0.80):
    """
    Motore NLP per rilevare video quasi-identici usando Cosine Similarity.
    Ritorna il dataframe originale con due nuove colonne: 'cluster_id' e 'is_cross_platform'.
    """
    df_overlap = df.copy()
    
    # 1. Normalizziamo i titoli
    df_overlap['norm_title'] = df_overlap['title'].apply(normalize_text)
    
    # Se mancano titoli, li riempiamo con stringhe vuote
    valid_docs = df_overlap['norm_title'].tolist()
    
    if len(valid_docs) < 2:
        df_overlap['cluster_id'] = df_overlap.index
        df_overlap['is_cross_platform'] = False
        return df_overlap

    # 2. Vettorizzazione (TF-IDF)
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(valid_docs)
    except:
        # Fallback se non ci sono abbastanza parole valide
        df_overlap['cluster_id'] = df_overlap.index
        df_overlap['is_cross_platform'] = False
        return df_overlap

    # 3. Calcolo Matrice di Similitudine del Coseno
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 4. Clustering (Algoritmo Greedy per raggruppare i nodi connessi)
    clusters = {} # Mappa indice_dataframe -> cluster_id
    cluster_counter = 0
    
    for i in range(len(cosine_sim)):
        if i not in clusters:
            # Crea un nuovo cluster
            clusters[i] = cluster_counter
            # Trova tutti i video simili a questo (sopra la soglia dell'80%)
            for j in range(i + 1, len(cosine_sim)):
                if cosine_sim[i][j] >= similarity_threshold:
                    clusters[j] = cluster_counter
            cluster_counter += 1
            
    # Assegniamo i cluster al DataFrame
    df_overlap['cluster_id'] = df_overlap.index.map(clusters)
    
    # 5. Determiniamo se un cluster è "Cross-Platform" o "Exclusive"
    # Un cluster è cross-platform se contiene video da più di 1 piattaforma
    cluster_platforms = df_overlap.groupby('cluster_id')['platform'].nunique()
    cross_platform_clusters = cluster_platforms[cluster_platforms > 1].index.tolist()
    
    df_overlap['is_cross_platform'] = df_overlap['cluster_id'].isin(cross_platform_clusters)
    
    # Aggiungiamo anche un tag per i "Duplicati Interni" (stesso video caricato 2 volte sulla stessa piattaforma)
    cluster_sizes = df_overlap.groupby('cluster_id').size()
    duplicate_clusters = cluster_sizes[cluster_sizes > 1].index.tolist()
    df_overlap['is_duplicate'] = df_overlap['cluster_id'].isin(duplicate_clusters)

    return df_overlap

def main():
    st.title("Cross-Platform Overlap & Duplication")
    st.markdown("""
    I creator riutilizzano i contenuti? In questa sezione un algoritmo di **Machine Learning (TF-IDF + Cosine Similarity)** 
    analizza la semantica dei titoli per rilevare video **"Near-Duplicate"** (quasi identici).
    Confrontiamo poi le performance dei video nativi (esclusivi) contro quelli riciclati su più piattaforme.
    """)

    with st.expander("💡 Come funziona l'algoritmo?"):
        st.markdown("""
        L'algoritmo raggruppa tutti i video che hanno una similarità testuale (titolo e metadati) superiore all'80%. Una volta creato un gruppo (Cluster), conta da quali piattaforme provengono i video all'interno di quel gruppo.
        
        * 🟠 **Duplicato Interno (Arancione):** Il cluster contiene più video, ma tutti provengono dalla stessa identica piattaforma (es. 2 video, entrambi da `youtube_shorts`). Questo indica il fenomeno dello spam o del reposting: un content creator carica lo stesso identico video più volte sul suo canale YouTube, magari in giorni diversi, sperando che almeno uno diventi virale.
        * 🟣 **Cross-Platform (Viola):** Il cluster contiene più video, e provengono da almeno due piattaforme diverse (es. 1 da `tiktok`, 1 da `instagram_reels`). Questo indica una strategia di distribuzione multi-canale: il creator crea un video e lo "ricicla", pubblicandolo identico su tutte le app per massimizzare il pubblico.
        * 🟢 **Esclusivo / Unico (Verde):** Il video non assomiglia a nessun altro nel dataset. È stato pubblicato una sola volta.
        """)

    with st.spinner("Ricerca duplicati semantici in corso..."):
        raw_df = get_data()
        
    if raw_df.empty:
        st.error("Nessun dato disponibile nel database.")
        return

    # Eseguiamo il motore di entity resolution
    df = detect_near_duplicates(raw_df)

    # Controlliamo lo stato attuale del dataset
    platforms_present = df['platform'].nunique() if 'platform' in df.columns else 0
    
    if platforms_present < 2:
        st.info("💡 **Modalità Standby:** Attualmente il database contiene solo dati da YouTube. L'algoritmo di rilevamento cross-platform è attivo, ma per individuare veri incroci tra piattaforme è necessario eseguire gli scraper di TikTok e Instagram. Nel frattempo, l'algoritmo cercherà duplicati interni (es. video ricaricati due volte).")

    # --- SEZIONE 1: Overview Duplicati ---
    st.header("1. Panoramica Contenuti Riciclati")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart dei contenuti unici vs duplicati/cross-platform
        df['Content Type'] = df.apply(
            lambda x: "Cross-Platform" if x['is_cross_platform'] 
            else ("Duplicato Interno" if x['is_duplicate'] else "Esclusivo (Unico)"), 
            axis=1
        )
        
        type_counts = df['Content Type'].value_counts().reset_index()
        type_counts.columns = ['Tipo', 'Conteggio']
        
        fig_pie = px.pie(type_counts, names='Tipo', values='Conteggio', hole=0.4,
                         title="Proporzione di Contenuti Originali vs Riciclati",
                         color='Tipo',
                         color_discrete_map={"Esclusivo (Unico)": "#00CC96", "Cross-Platform": "#AB63FA", "Duplicato Interno": "#FFA15A"})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Tabella dei cluster trovati
        dupes_df = df[df['is_duplicate'] == True]
        if not dupes_df.empty:
            st.write("📋 **Cluster di Video Simili Trovati:**")
            # Mostriamo i cluster raggruppati in modo leggibile
            display_dupes = dupes_df.groupby('cluster_id').agg({
                'title': list,
                'platform': list,
                'video_id': 'count'
            }).reset_index()
            display_dupes.columns = ['Cluster ID', 'Titoli Rilevati', 'Piattaforme Coinvolte', 'N. Video']
            st.dataframe(display_dupes[['N. Video', 'Piattaforme Coinvolte', 'Titoli Rilevati']], use_container_width=True)
        else:
            st.success("Tutti i video nel database attuale sembrano originali al 100%. Nessun duplicato rilevato.")

    st.divider()

    # --- SEZIONE 2: Engagement Comparison ---
    st.header("2. Analisi Comparativa dell'Engagement")
    st.markdown("I contenuti cross-platform generano più interazioni rispetto a quelli pubblicati su una singola piattaforma?")

    with st.expander("💡 Cosa ci dicono i risultati ?"):
        st.markdown("""
        * 🟢 **Video Esclusivi (Verde):** Hanno una mediana di 153.05k visualizzazioni e 5.7k like. Sono numeri molto buoni e rappresentano la normalità del dataset.
                    
        * 🟠 **Duplicati Interni (Arancione):** Le loro performance crollano (72.7k views, 2.2k like). Perché? Perché gli algoritmi di raccomandazione interni (come quello di YouTube o TikTok) riconoscono lo spam. Se un creator carica lo stesso video 3 volte, l'algoritmo smette di spingerlo nella homepage degli utenti, penalizzandone la visibilità.
                    
        * 🟣 **Cross-Platform (Viola):** Hanno metriche elevatissime (Mediana di 5.6 MILIONI di visualizzazioni e 272k like). Anche questo ha perfettamente senso logico. I video che i creator pubblicano su più piattaforme diverse sono solitamente i loro contenuti migliori, i "banger". Sono video progettati fin dall'inizio per essere virali. Inoltre, i grandi influencer (con milioni di follower) adottano sistematicamente questa strategia cross-platform, trascinando al rialzo la mediana del gruppo viola.
        """)

    if df['is_cross_platform'].any() or df['is_duplicate'].any():
        # Calcoliamo le mediane per tipo di contenuto
        eng_compare = df.groupby('Content Type')[['view_count', 'like_count']].median().reset_index()
        
        c1, c2 = st.columns(2)
        with c1:
            fig_views = px.bar(eng_compare, x='Content Type', y='view_count', text_auto=True,
                               title="Mediana Views: Originali vs Riciclati",
                               color='Content Type',
                               color_discrete_map={"Esclusivo (Unico)": "#00CC96", "Cross-Platform": "#AB63FA", "Duplicato Interno": "#FFA15A"})
            st.plotly_chart(fig_views, use_container_width=True)
            
        with c2:
            fig_likes = px.bar(eng_compare, x='Content Type', y='like_count', text_auto=True,
                               title="Mediana Mi Piace: Originali vs Riciclati",
                               color='Content Type',
                               color_discrete_map={"Esclusivo (Unico)": "#00CC96", "Cross-Platform": "#AB63FA", "Duplicato Interno": "#FFA15A"})
            st.plotly_chart(fig_likes, use_container_width=True)
    else:
        st.info("Statistiche di comparazione in attesa: appena l'algoritmo rileverà contenuti duplicati o cross-platform, i grafici di engagement comparato appariranno qui.")

if __name__ == "__main__":
    main()