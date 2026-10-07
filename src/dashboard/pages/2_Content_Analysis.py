import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import re
from pathlib import Path
from langdetect import detect, DetectorFactory
from collections import Counter
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from better_profanity import profanity

# Machine Learning & NLP
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import spacy

# Fissiamo il seed di langdetect per risultati riproducibili
DetectorFactory.seed = 0

# Importiamo la funzione di connessione
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data

st.set_page_config(page_title="Analisi del Contenuto", page_icon="🧠", layout="wide")

@st.cache_data
def detect_language(text):
    """Rileva la lingua depurando prima il testo dal rumore di Whisper."""
    if pd.isna(text) or not isinstance(text, str):
        return "Senza Testo / Muto"
        
    clean_t = text.strip().lower()
    # Rimuoviamo tag musicali o rumore comune prima di valutare la lingua
    clean_t = re.sub(r'\[.*?\]|\(.*?\)|♪|🎵', '', clean_t).strip()
    
    # Se dopo aver tolto la musica non resta quasi nulla, è un video muto
    if len(clean_t) < 10:
        return "Senza Testo / Muto"
        
    try:
        lang_code = detect(clean_t)
        if lang_code == 'en': return "Inglese"
        elif lang_code == 'it': return "Italiano"
        else: return "Altro"
    except:
        return "Altro"
    
# Nella categoria Altro sono presenti:
# - Trascrizioni di canzoni in altre lingue (es. spagnolo, coreano).
# - Testi estremamente brevi (es. "wow", "lol") che langdetect non riesce ad associare con confidenza all'inglese.
# - Rumori ambientali o parlato sovrapposto che Whisper ha tradotto in sequenze di testo incomprensibili

@st.cache_data
def compute_tfidf_keywords(df, text_col='clean_text', group_col='topic', top_n=10):
    """Calcola la TF-IDF con Fit Globale per estrarre parole dominanti specifiche per topic."""
    results = []
    
    # 1. Filtro Volgarità
    profanity.load_censor_words()
    swear_words = {str(word) for word in profanity.CENSOR_WORDSET}
    
    # 2. Rumore di Dominio ASR + Intercalari Comuni
    domain_stopwords = {
        'music', 'video', 'videos', 'tiktok', 'yeah', 'yes','okay', 'ok',
        'uh', 'um', 'hmm', 'huh', 'oh', 'hey', 'hi', 'hello',
        'like', 'love', 'just', 'got', 'gonna', 'wanna', 'baby', 'best', 'cause',
        'don', 'didn', 've', 'll', 're', 'isn', 'aren', 'wasn', 'weren',
        'think', 'know', 'going', 'want', 'right', 'said', 'feel', 'people', 
        'let', 'thank', 'thanks', 'way', 'time', 'good', 'look', 'make', 'say', 'really',
        'thing', 'things', 'today', 'tomorrow', 'guys', 'guy'
    }
    
    advanced_stopwords = list(spacy_stopwords.union(swear_words).union(domain_stopwords))
    
    # max_df=0.30 significa: ignora parole che compaiono in più del 30% dell'intero dataset
    vectorizer = TfidfVectorizer(
        stop_words=advanced_stopwords, 
        max_features=2000,
        ngram_range=(1, 2),
        token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',
        max_df=0.30 
    )
    
    # Prepariamo il corpus globale resettando gli indici per avere un allineamento perfetto
    df_valid = df.dropna(subset=[text_col, group_col]).reset_index(drop=True)
    full_corpus = df_valid[text_col].tolist()
    
    if not full_corpus: 
        return pd.DataFrame()
        
    try:
        # L'algoritmo ora impara il peso delle parole guardando TUTTI i macro-topic insieme
        tfidf_matrix = vectorizer.fit_transform(full_corpus)
        feature_names = vectorizer.get_feature_names_out()
    except ValueError:
        return pd.DataFrame()
        
    # 4. Estrazione dei Top Terms per ogni Topic
    for topic in df_valid[group_col].unique():
        # Troviamo gli indici di riga che corrispondono al topic corrente
        topic_indices = df_valid.index[df_valid[group_col] == topic].tolist()
        
        if not topic_indices: 
            continue
            
        # Ritagliamo la sottomatrice TF-IDF solo per i video di questo topic
        topic_matrix = tfidf_matrix[topic_indices]
        
        # Facciamo la media dei punteggi TF-IDF (asse 0 = colonne/parole)
        scores = topic_matrix.mean(axis=0).A1
        
        # Creiamo il dataframe e prendiamo le prime 'top_n' parole
        topic_scores = pd.DataFrame({'keyword': feature_names, 'score': scores})
        topic_scores = topic_scores.sort_values('score', ascending=False).head(top_n)
        topic_scores['topic'] = topic
        
        results.append(topic_scores)
        
    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

@st.cache_resource
def load_spacy_model():
    """Carica un modello spaCy più avanzato (Medium) per una maggiore precisione contestuale."""
    try:
        return spacy.load("en_core_web_md") # Cambiato da _sm a _md
    except OSError:
        st.error("Modello non trovato. Esegui nel terminale: python -m spacy download en_core_web_md")
        return None

@st.cache_data
def compute_ner(df, text_col='clean_text'):
    """Estrae Entità Nominate affidandosi al modello ML, con normalizzazione per il dominio social."""
    nlp = load_spacy_model()
    if not nlp: return pd.DataFrame()
    
    entities_data = []
    en_texts = df[df['lang'] == 'Inglese'][text_col].dropna().tolist()
    topics = df[df['lang'] == 'Inglese']['topic'].dropna().tolist()
    
    # Lista di "falsi positivi" generati dalle trascrizioni ASR (Whisper)
    noise_entities = [
        'one', 'two', 'first', 'today', 'tomorrow', 'music', 'music music', 'video', 'videos', 'yeah', 'tiktok', 'house', 'people', 'thing', 'things'
    ]
    
    max_docs = min(len(en_texts), 2000) 
    
    for i in range(max_docs):
        doc = nlp(en_texts[i])
        topic = topics[i]
        
        for ent in doc.ents:
            ent_text = ent.text.strip().title()
            ent_label = ent.label_
            
            # 1. Filtro Rumore: ignoriamo le entità troppo corte o presenti nella nostra blacklist
            if len(ent_text) < 3 or ent_text.lower() in noise_entities:
                continue
                
            # --- 2. LAYER DI NORMALIZZAZIONE E CONSOLIDAMENTO ---
            
            # Consolidamento "Trump": se la stringa contiene Trump, la unifichiamo e forziamo a PERSON
            if "Trump" in ent_text:
                ent_text = "Donald Trump"
                ent_label = "PERSON"
                
            # Consolidamento "Biden" 
            elif "Biden" in ent_text:
                ent_text = "Joe Biden"
                ent_label = "PERSON"
                
            # Intercettiamo le varianti comuni e le mappiamo a un'unica etichetta GPE (Geopolitical Entity)
            elif ent_text.lower() in ["u.s.", "u.s", "us", "usa", "u.s.a.", "united states", "the united states", "america"]:
                ent_text = "United States"
                ent_label = "GPE"
                
            # Intercettiamo eventuali altri falsi positivi musicali sfuggiti al controllo esatto
            elif "Music" in ent_text:
                continue

            # 3. Salvataggio finale
            if ent_label in ['PERSON', 'ORG', 'GPE']:
                entities_data.append({
                    'entity': ent_text,
                    'label': ent_label,
                    'topic': topic
                })
                
    return pd.DataFrame(entities_data)

def main():
    st.title("Content & Text Analysis")
    st.markdown("""
    In questa sezione applichiamo tecniche di **Natural Language Processing (NLP)** sulle trascrizioni ASR dei video.
    Analizziamo la distribuzione linguistica, la gerarchia semantica dei topic e i pattern lessicali (TF-IDF e NER) 
    per comprendere a fondo di cosa si parla nei video.
    """)

    with st.spinner("Caricamento e analisi testi in corso..."):
        df = get_data()

    if df.empty or 'transcript' not in df.columns:
        st.error("Nessun dato di trascrizione disponibile.")
        return

    # Prepariamo la colonna testo per tutti i 1000 video
    df['clean_text'] = df['transcript'].fillna("").astype(str).str.strip()
    
    # Applichiamo la rilevazione lingua a TUTTO il dataframe (per non perdere i 216 video)
    df['lang'] = df['clean_text'].apply(detect_language)

    # --- SEZIONE 1: Demografia Linguistica Globale ---
    st.header("1. Distribuzione Linguistica Globale")
    
    # Ora conterà esattamente 1000 video
    lang_counts = df['lang'].value_counts().reset_index()
    lang_counts.columns = ['Lingua', 'Numero di Video']
    
    fig_lang = px.bar(
        lang_counts, x='Lingua', y='Numero di Video', color='Lingua', text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_lang.update_layout(showlegend=False)
    st.plotly_chart(fig_lang, use_container_width=True)

    st.divider()

    # --- FILTRO PER LE SEZIONI SUCCESSIVE (TF-IDF e NER) ---
    # Per NLP e Topic Modeling, scartiamo i video muti che farebbero crashare i modelli
    valid_texts_df = df[df['lang'] != "Senza Testo / Muto"].copy()

    if valid_texts_df.empty:
        st.warning("Tutti i video nel database sembrano avere trascrizioni vuote.")
        return

    # --- SEZIONE 2: Classificazione Semantica Gerarchica ---
    st.header("2. Classificazione Semantica Gerarchica")
    st.markdown("Composizione dei Macro-Topic in base ai Sotto-Topic estratti dall'Intelligenza Artificiale (Llama-3).")
    
    # Tassonomia fissa (dal file dynamic_taxonomy.json)
    dynamic_taxonomy = {
        "finance": ["Cryptocurrency", "Trading & Investing", "Personal Finance & Budgeting", "Real Estate & Business", "Macroeconomics"],
        "nutrition": ["High-Protein & Muscle", "Weight Loss & Fasting", "Vegan & Plant-Based", "General Health & Habits", "Supplements"],
        "politics": ["Elections & Campaigns", "International Relations", "Domestic Policy & Economy", "Social Issues & Debates", "Political Satire"]
    }

    if 'llm_topic_groq' in df.columns and 'topic' in df.columns:
        bar_df = df.dropna(subset=['llm_topic_groq']).copy()
        
        if not bar_df.empty:
            # Raggruppamento dati
            topic_data = bar_df.groupby(['topic', 'llm_topic_groq']).size().reset_index(name='count')
            
            # Creiamo 3 colonne in Streamlit per affiancare i grafici
            col1, col2, col3 = st.columns(3)
            
            # Prepariamo la lista dei macro-topic e le colonne corrispondenti
            macro_topics = ["finance", "nutrition", "politics"]
            columns = [col1, col2, col3]
            
            # Palette di colori differenziate per rendere i grafici unici
            color_scales = [px.colors.sequential.Teal, px.colors.sequential.Oranges, px.colors.sequential.Purples]

            for i, macro_topic in enumerate(macro_topics):
                with columns[i]:
                    st.subheader(macro_topic.capitalize())
                    
                    # Filtriamo solo i dati del macro-topic corrente
                    df_sub = topic_data[topic_data['topic'] == macro_topic].copy()
                    
                    # Assicuriamoci di mappare correttamente l'ordine rispetto alla tassonomia
                    if not df_sub.empty:
                        fig_pie = px.pie(
                            df_sub, 
                            values='count', 
                            names='llm_topic_groq', 
                            hole=0.4, # Stile "Donut" (Ciambella) molto più moderno
                            color_discrete_sequence=color_scales[i]
                        )
                        
                        # Spostiamo la legenda in basso e ottimizziamo gli spazi
                        fig_pie.update_layout(
                            legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
                            margin=dict(t=10, b=10, l=10, r=10)
                        )
                        
                        # Mostriamo sia la percentuale che il valore assoluto nel grafico
                        fig_pie.update_traces(textposition='inside', textinfo='percent+value')
                        
                        st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.info(f"Nessun dato per {macro_topic}")
            
            # Manteniamo l'expander con la tabella grezza per i secchioni dei dati
            with st.expander("Dettaglio Analitico Completo dei Sotto-Topic"):
                st.dataframe(topic_data.sort_values(['topic', 'count'], ascending=[True, False]), use_container_width=True, hide_index=True)
    
    st.divider()

    # --- SEZIONE 3: Keyword Extraction (TF-IDF) ---
    st.header("3. Keyword Extraction (TF-IDF)")
    st.markdown("""
    Quali sono le parole chiave che **distinguono** maggiormente un dominio dall'altro? 
    Utilizzando l'algoritmo TF-IDF, penalizziamo le parole comuni ed estraiamo il lessico caratteristico di ogni Macro-Topic.
    """)
    
    with st.spinner("Calcolo TF-IDF in corso..."):
        tfidf_df = compute_tfidf_keywords(valid_texts_df[valid_texts_df['lang'] == 'Inglese'])
        
    if not tfidf_df.empty:
        fig_tfidf = px.bar(
            tfidf_df, x='score', y='keyword', color='topic', facet_col='topic',
            orientation='h', text_auto='.2f',
            labels={'score': 'Importanza (TF-IDF)', 'keyword': ''},
            title="Top 10 Keyword per Dominio",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_tfidf.update_yaxes(matches=None, showticklabels=True)
        fig_tfidf.update_layout(showlegend=False)
        st.plotly_chart(fig_tfidf, use_container_width=True)
    else:
        st.info("Non ci sono abbastanza dati in inglese per estrarre le keyword.")

    st.divider()

    # --- SEZIONE 4: Named Entity Recognition (NER) ---
    st.header("4. Named Entity Recognition (NER)")
    st.markdown("""
    Estraiamo soggetti reali dalle trascrizioni: **Persone (PERSON), Organizzazioni (ORG) e Luoghi (GPE)**. 
    Questo ci permette di scoprire chi e cosa domina le conversazioni sulle piattaforme social.
    """)
    
    with st.spinner("Estrazione Entità Nominate (spaCy) in corso..."):
        ner_df = compute_ner(valid_texts_df)
        
    if not ner_df.empty:
        # Contiamo le entità più frequenti
        top_entities = ner_df.groupby(['label', 'entity', 'topic']).size().reset_index(name='count')
        top_entities = top_entities.sort_values('count', ascending=False).groupby('label').head(10)
        
        # Mappa i nomi delle label per maggiore chiarezza
        label_map = {'PERSON': '👤 Persone', 'ORG': '🏢 Organizzazioni/Aziende', 'GPE': '🌍 Luoghi/Nazioni'}
        top_entities['label_name'] = top_entities['label'].map(label_map)
        
        fig_ner = px.treemap(
            top_entities, path=[px.Constant("Entità Nominate"), 'label_name', 'entity'], values='count',
            color='topic', color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Mappa delle Entità più citate (Dimensioni = Frequenza)"
        )
        fig_ner.update_traces(root_color="lightgrey")
        st.plotly_chart(fig_ner, use_container_width=True)
        
        with st.expander("Esplora la tabella delle Entità"):
            st.dataframe(ner_df['entity'].value_counts().reset_index().head(50), use_container_width=True)
    else:
        st.info("Impossibile estrarre entità dai testi correnti.")

if __name__ == "__main__":
    main()