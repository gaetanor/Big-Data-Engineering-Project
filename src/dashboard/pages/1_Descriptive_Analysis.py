import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path

# Aggiungiamo la root al path per poter importare utils
# (Streamlit lancia le pages da una directory diversa, questo garantisce che trovi utils.py)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data

st.set_page_config(page_title="Analisi Descrittiva", page_icon="📊", layout="wide")

def main():
    st.title("Analisi Descrittiva Cross-Platform")
    st.markdown("""
    Questa sezione esplora la composizione del dataset, confrontando i **volumi di pubblicazione**, 
    i **trend temporali** e la distribuzione delle **metriche di engagement** tra le diverse piattaforme e i domini tematici.
    """)

    # 1. Caricamento e preparazione dati
    with st.spinner("Caricamento dati..."):
        df = get_data()

    if df.empty:
        st.warning("Nessun dato disponibile per l'analisi.")
        return

    # --- SIDEBAR: Filtri Interattivi ---
    st.sidebar.header("⚙️ Filtri Analisi")
    
    platforms = df['platform'].unique() if 'platform' in df.columns else []
    topics = df['topic'].unique() if 'topic' in df.columns else []
    
    selected_platform = st.sidebar.multiselect("Piattaforme", platforms, default=platforms)
    selected_topic = st.sidebar.multiselect("Topic (Domini)", topics, default=topics)

    # Applicazione filtri
    filtered_df = df.copy()
    if selected_platform:
        filtered_df = filtered_df[filtered_df['platform'].isin(selected_platform)]
    if selected_topic:
        filtered_df = filtered_df[filtered_df['topic'].isin(selected_topic)]

    if filtered_df.empty:
        st.error("I filtri selezionati non hanno prodotto alcun risultato.")
        return

    # --- SEZIONE 1: Volumi di Contenuto ---
    st.header("1. Volumi di Contenuto")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Video per Piattaforma e Topic")
        # Raggruppiamo i dati
        vol_df = filtered_df.groupby(['platform', 'topic']).size().reset_index(name='count')
        fig_vol = px.bar(
            vol_df, x='platform', y='count', color='topic', 
            barmode='group', text_auto=True,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={'platform': 'Piattaforma', 'count': 'Numero di Video', 'topic': 'Dominio Tematico'}
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        st.subheader("Durata Media dei Video")
        # Assicuriamoci che la colonna duration sia numerica
        filtered_df['duration'] = pd.to_numeric(filtered_df.get('duration', 0), errors='coerce').fillna(0)
        
        # Calcoliamo la media ignorando i video con durata 0
        dur_df = filtered_df[filtered_df['duration'] > 0].groupby(['topic', 'platform'])['duration'].mean().reset_index(name='avg_duration')
        
        if not dur_df.empty:
            fig_dur = px.bar(
                dur_df, x='topic', y='avg_duration', color='platform', 
                barmode='group', text_auto='.1f',
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'topic': 'Dominio Tematico', 'avg_duration': 'Durata Media (s)', 'platform': 'Piattaforma'}
            )
            fig_dur.update_layout(yaxis_title="Secondi")
            st.plotly_chart(fig_dur, use_container_width=True)
        else:
            st.info("Dati sulla durata non disponibili.")


    st.divider()

    # --- SEZIONE 2: Distribuzione Temporale ---
    st.header("2. Distribuzione Temporale delle Pubblicazioni")
    
    if 'upload_date' in filtered_df.columns:
        # Rimuoviamo i valori nulli per le date
        time_df = filtered_df.dropna(subset=['upload_date']).copy()
        
        if not time_df.empty:
            # Creiamo una colonna con l'Anno-Mese per aggregare i dati
            time_df['year_month'] = time_df['upload_date'].dt.to_period('M').astype(str)
            trend_df = time_df.groupby(['year_month', 'platform']).size().reset_index(name='video_count')
            
            # Ordiniamo temporalmente
            trend_df = trend_df.sort_values('year_month')

            fig_time = px.line(
                trend_df, x='year_month', y='video_count', color='platform',
                markers=True,
                labels={'year_month': 'Anno di Pubblicazione', 'video_count': 'Video Pubblicati', 'platform': 'Piattaforma'},
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("Nessun dato temporale valido per la tracciatura della serie storica.")
    else:
        st.info("La colonna 'upload_date' non è presente nel dataset.")


    st.divider()

    # --- SEZIONE 3: Analisi dell'Engagement ---
    st.header("3. Distribuzione dell'Engagement")
    st.markdown("""
    *Nota: I grafici dell'engagement utilizzano una **scala logaritmica** sull'asse Y per gestire la natura estrema ("Power Law") delle metriche social, permettendo di visualizzare correttamente sia i video di nicchia che quelli virali.*
    """)

    # Selector per scegliere la metrica da visualizzare (interattività aggiuntiva)
    metric_map = {
        'Visualizzazioni': 'view_count',
        'Mi Piace': 'like_count',
        'Commenti': 'comment_count'
    }
    selected_metric_label = st.radio("Seleziona la metrica da analizzare:", list(metric_map.keys()), horizontal=True)
    metric_col = metric_map[selected_metric_label]

    if metric_col in filtered_df.columns:
        # Box plot per analizzare mediane e outlier
        fig_eng = px.box(
            filtered_df, x='platform', y=metric_col, color='topic',
            log_y=True, # SCALA LOGARITMICA FONDAMENTALE
            points="all", # Mostra anche i singoli punti a lato del box
            labels={'platform': 'Piattaforma', metric_col: selected_metric_label, 'topic': 'Dominio'},
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        st.plotly_chart(fig_eng, use_container_width=True)
    else:
        st.warning(f"La metrica {selected_metric_label} non è disponibile nel dataset attuale.")

if __name__ == "__main__":
    main()