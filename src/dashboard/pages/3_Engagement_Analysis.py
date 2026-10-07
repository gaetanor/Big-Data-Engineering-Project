import streamlit as st
import pandas as pd
import plotly.express as px
import sys
from pathlib import Path
from textblob import TextBlob
import numpy as np

# Importiamo la funzione di connessione
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data

st.set_page_config(page_title="Analisi dell'Engagement", page_icon="🔥", layout="wide")

@st.cache_data
def engineer_features(df):
    df_eng = df.copy()
    
    df_eng['duration_sec'] = pd.to_numeric(df_eng['duration'], errors='coerce').fillna(0)
    
    def count_tags(x):
        if isinstance(x, list): return len(x)
        if isinstance(x, str): return len(x.split(','))
        return 0
    df_eng['hashtag_count'] = df_eng['hashtags'].apply(count_tags)
    
    # 3a. Sentiment del CREATORE (Trascrizione del video)
    def get_sentiment(text):
        if not text or not isinstance(text, str): return 0.0
        return TextBlob(text).sentiment.polarity
        
    df_eng['sentiment_score'] = df_eng['transcript'].apply(get_sentiment)
    
    # --- PRIMA MODIFICA: Soglie più sensibili per il Creatore ---
    df_eng['sentiment_category'] = pd.cut(
        df_eng['sentiment_score'], 
        bins=[-1.0, -0.05, 0.05, 1.0], # Fascia di neutralità ristretta al 5%
        labels=['Negativo', 'Neutro', 'Positivo'],
        include_lowest=True
    )

    # 3b. Sentiment dell'AUDIENCE (Testo dei commenti)
    def get_comments_sentiment(comments_data):
        if not isinstance(comments_data, list) or len(comments_data) == 0:
            return 0.0
        
        scores = []
        for c in comments_data:
            text = c.get('text', '')
            if text and isinstance(text, str):
                scores.append(TextBlob(text).sentiment.polarity)
                
        # NOVITÀ: Rimuoviamo gli zero assoluti per non annacquare i sentimenti forti
        active_scores = [s for s in scores if s != 0.0]
        
        return sum(active_scores) / len(active_scores) if active_scores else 0.0

    if 'comments' in df_eng.columns:
        df_eng['audience_sentiment_score'] = df_eng['comments'].apply(get_comments_sentiment)
        
        # Fasce ultra-sensibili
        df_eng['audience_sentiment_category'] = pd.cut(
            df_eng['audience_sentiment_score'], 
            bins=[-1.0, -0.01, 0.01, 1.0], 
            labels=['Negativo', 'Neutro', 'Positivo'],
            include_lowest=True
        )
    else:
        df_eng['audience_sentiment_score'] = 0.0
        df_eng['audience_sentiment_category'] = 'Neutro'
    
    return df_eng

def main():
    st.title("Analisi dell'Engagement e Correlazioni")
    st.markdown("""
    Cosa rende virale un video? In questa sezione analizziamo la relazione matematica tra le caratteristiche del contenuto 
    (**lunghezza, numero di hashtag, sentiment del parlato**) e le metriche di popolarità (**Visualizzazioni e Mi Piace**).
    """)

    with st.spinner("Calcolo delle correlazioni e del sentiment in corso..."):
        raw_df = get_data()
        
    if raw_df.empty:
        st.error("Nessun dato disponibile nel database.")
        return

    # Applichiamo l'ingegneria delle feature
    df = engineer_features(raw_df)

    # --- SEZIONE 1: Matrice di Correlazione (Spearman) ---
    st.header("1. Matrice di Correlazione")
    st.markdown("""
    *Nota metodologica: Utilizziamo il coefficiente di correlazione di **Spearman** (basato sui ranghi) 
    invece di Pearson, poiché le metriche social non seguono una distribuzione normale ma una Power Law.*
    """)
    
    # Selezioniamo solo le colonne numeriche rilevanti
    corr_cols = ['duration_sec', 'hashtag_count', 'sentiment_score', 'view_count', 'like_count', 'comment_count']
    available_cols = [c for c in corr_cols if c in df.columns]
    
    if len(available_cols) > 1:
        # Calcolo della matrice di correlazione
        corr_matrix = df[available_cols].corr(method='spearman')
        
        fig_corr = px.imshow(
            corr_matrix, 
            text_auto=".2f", 
            aspect="auto",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1,
            title="Mappa di calore delle correlazioni"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Metriche insufficienti per calcolare la correlazione.")

    st.divider()

    # --- SEZIONE 2: Scatter Plot Interattivi (Regressione OLS) ---
    st.header("2. Analisi Multivariata")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Configura Analisi")
        x_axis = st.selectbox("Variabile Indipendente (Causa):", 
                              ['duration_sec', 'hashtag_count', 'sentiment_score'], 
                              format_func=lambda x: {'duration_sec': 'Durata (Secondi)', 'hashtag_count': 'Numero di Hashtag', 'sentiment_score': 'Polarità Sentiment'}[x])
                              
        y_axis = st.selectbox("Variabile Dipendente (Effetto):", 
                              ['view_count', 'like_count', 'comment_count'],
                              format_func=lambda x: {'view_count': 'Visualizzazioni', 'like_count': 'Mi Piace', 'comment_count': 'Commenti'}[x])
                              
        color_by = st.selectbox("Raggruppa per:", ['topic', 'platform', 'sentiment_category'])

    with col2:
        # 1. Prepariamo i dati
        plot_df = df.copy()
        plot_df[y_axis] = plot_df[y_axis] + 1 
        
        # 2. SANITIZZAZIONE DELLE STRINGHE (La cura per l'errore JSON)
        # Rimuoviamo doppi apici, ritorni a capo e backslash dai titoli
        plot_df['title'] = plot_df['title'].astype(str) \
            .str.replace('"', "'", regex=False) \
            .str.replace('\n', ' ', regex=False) \
            .str.replace('\r', '', regex=False) \
            .str.replace('\\', '', regex=False)
        
        # 3. PULIZIA MATEMATICA CRITICA
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        plot_df.dropna(subset=[x_axis, y_axis, color_by, 'title', 'video_id'], inplace=True)
        plot_df = plot_df[plot_df[y_axis] > 0]
        
        # 4. Creazione del grafico
        if not plot_df.empty:
            # Plotly Scatter con linea di tendenza OLS
            fig_scatter = px.scatter(
                plot_df, x=x_axis, y=y_axis, color=color_by,
                log_y=True, 
                trendline="ols", 
                trendline_scope="overall", 
                hover_data=['title', 'video_id'],
                title=f"Impatto di {x_axis} su {y_axis} (Scala Logaritmica)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("Attenzione: Non ci sono dati validi sufficienti per generare questo grafico.")
    st.divider()

    # --- SEZIONE 3: Sentiment dell'Audience & Engagement ---
    st.header("3. Relazione tra Opinione del Pubblico ed Engagement")
    st.markdown("""
    Questa sezione analizza in che modo il feedback degli utenti (estratto dai commenti) influenzi le metriche di popolarità. 
    Il sentiment dei commenti è calcolato in relazione ai contenuti del video e classificato in tre categorie: **Positivo, Neutro o Negativo**.
    
    *Nota metodologica: Utilizziamo le mediane al posto delle medie per evitare che singoli video estremamente virali distorcano la rappresentazione grafica complessiva.*
    """)
    
    # Costringiamo Plotly a mostrare sempre tutti e tre i colori/categorie
    cat_orders = {"audience_sentiment_category": ["Negativo", "Neutro", "Positivo"]}
    color_map = {'Negativo': '#EF553B', 'Neutro': '#00CC96', 'Positivo': '#636EFA'}
    
    # Aggreghiamo calcolando le mediane per Views, Likes e Comment_Count (ora incluso)
    audience_agg = df.groupby('audience_sentiment_category', observed=False)[['view_count', 'like_count', 'comment_count']].median().reset_index()
    
    # Layout a 3 colonne per affiancare Views, Likes e Comments
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        fig_aud_views = px.bar(
            audience_agg, x='audience_sentiment_category', y='view_count',
            color='audience_sentiment_category',
            title="Mediana Visualizzazioni",
            text_auto=True,
            color_discrete_map=color_map,
            category_orders=cat_orders
        )
        fig_aud_views.update_layout(xaxis_title="Sentiment Audience", yaxis_title="Views", showlegend=False)
        st.plotly_chart(fig_aud_views, use_container_width=True)

    with col_a2:
        fig_aud_likes = px.bar(
            audience_agg, x='audience_sentiment_category', y='like_count',
            color='audience_sentiment_category',
            title="Mediana Mi Piace",
            text_auto=True,
            color_discrete_map=color_map,
            category_orders=cat_orders
        )
        fig_aud_likes.update_layout(xaxis_title="Sentiment Audience", yaxis_title="Likes", showlegend=False)
        st.plotly_chart(fig_aud_likes, use_container_width=True)
        
    with col_a3:
        fig_aud_comments = px.bar(
            audience_agg, x='audience_sentiment_category', y='comment_count',
            color='audience_sentiment_category',
            title="Mediana Commenti",
            text_auto=True,
            color_discrete_map=color_map,
            category_orders=cat_orders
        )
        fig_aud_comments.update_layout(xaxis_title="Sentiment Audience", yaxis_title="Commenti", showlegend=False)
        st.plotly_chart(fig_aud_comments, use_container_width=True)

    st.divider()

    # --- SEZIONE 4: Analisi dei Creator (Popolarità e Sentiment) ---
    st.divider()
    st.header("4. Classifica Creator")
    st.markdown("Questa tabella individua i creatori più influenti, quelli più amati e quelli più controversi.")

    if 'uploader' in df.columns:
        # Raggruppiamo i dati per creatore e piattaforma
        creator_df = df.groupby(['uploader', 'platform']).agg(
            total_views=('view_count', 'sum'),
            avg_sentiment=('audience_sentiment_score', 'mean'),
            video_count=('video_id', 'count')
        ).reset_index()

        # Puliamo i creator senza nome
        creator_df = creator_df[creator_df['uploader'].str.strip() != ""]

        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.subheader("Più Popolari")
            top_pop = creator_df.sort_values('total_views', ascending=False).head(5)
            st.dataframe(top_pop[['uploader', 'platform', 'total_views']], hide_index=True, use_container_width=True)
            
        with col_c2:
            st.subheader("Più Amati")
            # Filtriamo chi ha almeno un sentiment positivo per evitare i neutri in cima
            top_loved = creator_df[creator_df['avg_sentiment'] > 0].sort_values('avg_sentiment', ascending=False).head(5)
            st.dataframe(top_loved[['uploader', 'platform', 'avg_sentiment']], hide_index=True, use_container_width=True)
            
        with col_c3:
            st.subheader("Più Controversi")
            # Filtriamo per prendere i sentimenti più negativi
            top_hated = creator_df[creator_df['avg_sentiment'] < 0].sort_values('avg_sentiment', ascending=True).head(5)
            if top_hated.empty:
                st.info("Nessun creator con sentiment mediamente negativo rilevato.")
            else:
                st.dataframe(top_hated[['uploader', 'platform', 'avg_sentiment']], hide_index=True, use_container_width=True)
    else:
        st.warning("Colonna 'uploader' mancante nel dataset.")


    # --- SEZIONE 5: Sentiment Cross-Platform per Topic ---
    st.divider()
    st.header("5. Impatto del Topic sul Pubblico in base alla Piattaforma")
    st.markdown("Questa Heatmap mostra se lo stesso argomento viene accolto in modo diverso a seconda della piattaforma (es. Finance su TikTok vs Instagram). I colori rossi indicano negatività, i blu positività.")

    if 'topic' in df.columns and 'platform' in df.columns:
        # Creiamo una tabella pivot (Media del sentiment per Topic/Piattaforma)
        sentiment_pivot = df.pivot_table(
            values='audience_sentiment_score', 
            index='topic', 
            columns='platform', 
            aggfunc='mean'
        ).fillna(0)

        fig_heat = px.imshow(
            sentiment_pivot,
            text_auto=".3f", # Mostra il valore con 3 decimali
            aspect="auto",
            color_continuous_scale="RdBu", # Rosso per negativo, Blu per positivo
            zmin=-0.3, zmax=0.3, # Centra lo zero cromaticamente
            title="Sentiment Medio dell'Audience: Topic vs Piattaforma",
            labels=dict(x="Piattaforma", y="Dominio Tematico", color="Sentiment Score")
        )
        st.plotly_chart(fig_heat, use_container_width=True)

if __name__ == "__main__":
    main()