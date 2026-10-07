import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import json
from pathlib import Path
from chatbot_logic import generate_video_explanation

# Aggiunta al path per poter importare utils
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import get_data

st.set_page_config(page_title="Misinformation Analysis", page_icon="⚖️", layout="wide")

# --- PATHS ---
DATA_LAKE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data_lake"
CONSENSUS_FILE = DATA_LAKE_DIR / "final_consensus_dataset.csv"
METRICS_FILE = DATA_LAKE_DIR / "agreement_metrics.json"
LOCAL_ANNOTATIONS_DIR = DATA_LAKE_DIR / "local_annotations"

# Colori per i grafici
COLOR_MAP = {
    'reliable': '#00CC96', 'misleading': '#FFA15A', 
    'out-of-context': '#FECB52', 'satire': '#AB63FA', 
    'false': '#EF553B', 'ambiguous': '#B6E880'
}

def main():
    st.title("Analisi dell'Affidabilità dei Contenuti")
    
    with st.spinner("Caricamento metadati e annotazioni..."):
        raw_df = get_data()
        
    if raw_df.empty or not CONSENSUS_FILE.exists():
        st.error("Dati mancanti. Assicurati di aver generato il dataset finale in MongoDB e nel Data Lake.")
        return

    # Unione dati originali
    consensus_df = pd.read_csv(CONSENSUS_FILE)
    raw_df['video_id'] = raw_df['video_id'].astype(str)
    consensus_df['video_id'] = consensus_df['video_id'].astype(str)
    df = pd.merge(raw_df, consensus_df, on='video_id', how='inner')

    # Creazione dei 3 Tabs (Le sottopagine)
    tab1, tab2, tab3 = st.tabs([
        "1. Majority Vote", 
        "2. Weighted Vote", 
        "3. Local Models"
    ])

    # ==========================================
    # TAB 1: MAJORITY VOTE
    # ==========================================
    with tab1:
        st.header("Metodo 1: Voto a Maggioranza")
        st.markdown("""
            In questa sezione valutiamo la veridicità dei contenuti. 
            Ogni video è stato esaminato da una giuria indipendente di 5 Intelligenze Artificiali (GPT-4o, Qwen-3-235B-A22B-2507, Llama-3.3-70B-Instruct, DeepSeek-V4-Flash, Gemini-3.1-Flash-Lite).
            Ogni modello AI ha esattamente 1 voto. Vince la maggioranza.
        """)

        st.subheader("Accordo tra i Giudici")
        if METRICS_FILE.exists():
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            alpha = metrics_data.get("krippendorff_alpha")
            if alpha:
                st.metric("Krippendorff's Alpha", f"{alpha:.3f}", metrics_data.get("alpha_evaluation", ""), delta_color="inverse")
            
            kappa_data = metrics_data.get("cohen_kappa", {})
            if kappa_data:
                kappa_df = pd.DataFrame(list(kappa_data.items()), columns=["Coppia", "Cohen's Kappa"]).sort_values(by="Cohen's Kappa", ascending=False)
                st.dataframe(kappa_df.style.background_gradient(cmap='Blues', subset=["Cohen's Kappa"]), use_container_width=True, hide_index=True)
                
                with st.expander("📖 Analisi dei Risultati"):
                    st.markdown("""
                    **Perché l'accordo è basso?**
                    Un Alpha di 0.356 non è un errore tecnico, ma una dimostrazione dell'ambiguità del dominio. 
                    Mentre i modelli "pesanti" (GPT-4o, Llama 70B) mostrano un buon allineamento, i modelli "Lite" introducono forte rumore statistico.
                    Questo giustifica l'uso di una giuria al posto di un singolo modello, ma suggerisce che il voto paritario penalizza le intelligenze più avanzate.
                    """)
        
        fig_dist = px.pie(
            df['consensus_label'].value_counts().reset_index(name='count'), 
            values='count', names='consensus_label', hole=0.4, 
            color='consensus_label', color_discrete_map=COLOR_MAP,
            title="Distribuzione Finale"
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.divider()
        st.header("Ispezione e Spiegazione AI")
        search = st.text_input("Cerca per titolo del video:")
        
        display_cols = ['title', 'platform', 'consensus_label', 'weighted_consensus']
        if 'weighted_consensus' not in df.columns:
            df['weighted_consensus'] = "N/A"
            
        vote_cols = [c for c in df.columns if c.startswith('vote_')]
        display_cols.extend(vote_cols)
        
        filtered_df = df.copy()
        if search:
            filtered_df = filtered_df[filtered_df['title'].str.contains(search, case=False, na=False)]
            
        selection_event = st.dataframe(
            filtered_df[display_cols], 
            use_container_width=True, on_select="rerun", selection_mode="single-row"
        )

        selected_rows = selection_event.selection.rows
        if len(selected_rows) > 0:
            selected_index = selected_rows[0]
            selected_row = filtered_df.iloc[selected_index]
            vid_id = str(selected_row['video_id'])
            
            st.subheader(f"Dettagli: {selected_row['title']}")
            if st.button("Genera Spiegazione con AI", type="primary"):
                with st.spinner("Sintesi dei verdetti in corso..."):
                    annotation_path = DATA_LAKE_DIR / "annotations" / f"{vid_id}_annotation.json"
                    if annotation_path.exists():
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            annotation_data = json.load(f)
                        
                        metadata = {
                            "title": selected_row['title'], "platform": selected_row['platform'],
                            "consensus_label": selected_row['consensus_label']
                        }
                        try:
                            explanation = generate_video_explanation(vid_id, metadata, annotation_data)
                            st.info(f"**Sintesi:**\n\n{explanation}")
                        except Exception as e:
                            st.error(f"Errore RAG: {e}")
                    else:
                        st.error("JSON di annotazione non trovato per questo video.")

    # ==========================================
    # TAB 2: WEIGHTED VOTE
    # ==========================================
    with tab2:
        st.header("Metodo 2: Voto Ponderato")
        st.markdown("Invece di assegnare 1 voto per modello, assegniamo un peso proporzionale alla dimensione (capacità di ragionamento) della rete neurale.")
        
        # Formule Matematiche Renderizzate in UI
        st.latex(r"\text{Final Label} = \arg\max_L \sum_{i \in \text{Judges}} w_i \cdot \mathbb{I}(\text{Judge}_i = L)")
        st.markdown("Per evitare che i modelli massicci azzerino il voto dei più piccoli, utilizziamo una **scala logaritmica**:")
        st.latex(r"w_i = \log_{10}(\text{Parameters in Billions}) + 1")
        
        # Schema dei Pesi Calcolati
        weights = {
            'vote_Judge_GPT_4o': 4.00,       # Est. ~1000B
            'vote_Judge_Qwen_3': 3.37,       # 235B
            'vote_Judge_Llama_70B': 2.85,    # 70B
            'vote_Judge_DeepSeek': 2.18,     # Est. ~15B (Flash)
            'vote_Judge_Gemini-3.1': 1.90    # Est. ~8B (Lite)
        }
        
        # Mostriamo i pesi nella UI in modo elegante
        st.markdown("### Pesi Assegnati ai Modelli")
        cols = st.columns(len(weights))
        model_display_names = ["GPT-4o (~1000B)", "Qwen 3 (235B)", "Llama 3 (70B)", "DeepSeek (~15B)", "Gemini Lite (~8B)"]
        for i, (model_key, w) in enumerate(weights.items()):
            cols[i].metric(model_display_names[i], f"x {w:.2f}")
            
        # Calcolo dinamico del nuovo consenso
        def calculate_weighted_consensus(row):
            scores = {}
            for col, weight in weights.items():
                if col in row and pd.notna(row[col]) and row[col] != 'error':
                    label = row[col]
                    scores[label] = scores.get(label, 0.0) + weight
            if not scores: return 'ambiguous'
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]

        df['weighted_consensus'] = df.apply(calculate_weighted_consensus, axis=1)
            
        st.divider()
        
        fig_w_dist = px.pie(
            df['weighted_consensus'].value_counts().reset_index(name='count'), 
            values='count', names='weighted_consensus', hole=0.4, 
            color='weighted_consensus', color_discrete_map=COLOR_MAP,
            title="Distribuzione Finale"
        )
        st.plotly_chart(fig_w_dist, use_container_width=True)

    # ==========================================
    # TAB 3: LOCAL MODELS
    # ==========================================
    with tab3:
        st.header("Metodo 3: Modelli Locali")
        st.markdown("Questa sezione aggrega i verdetti emessi da modelli open-source eseguiti in locale (Llama-3.2-3B, Mistral-7.2B, Gemma4:e2b-5B, Qwen3-8.2B, DeepSeek:r1-8B).")
        
        if not LOCAL_ANNOTATIONS_DIR.exists():
            st.warning(f"La cartella `{LOCAL_ANNOTATIONS_DIR.name}` non esiste ancora. Assicurati di aver eseguito i modelli locali e salvato le annotazioni.")
        else:
            local_files = list(LOCAL_ANNOTATIONS_DIR.glob("*.json"))
            if not local_files:
                st.info("La cartella è presente ma vuota.")
            else:
                local_data = []
                for f in local_files:
                    try:
                        with open(f, 'r', encoding='utf-8') as j_file:
                            data = json.load(j_file)
                            local_data.append({
                                "video_id": data.get("video_id"),
                                "local_consensus": data.get("consensus_label", "error")
                            })
                    except Exception:
                        pass
                
                local_df = pd.DataFrame(local_data)
                local_df['video_id'] = local_df['video_id'].astype(str)
                merged_local = pd.merge(raw_df, local_df, on='video_id', how='inner')
                
                if not merged_local.empty:
                    fig_local = px.pie(
                        merged_local['local_consensus'].value_counts().reset_index(name='count'), 
                        values='count', names='local_consensus', hole=0.4, 
                        color='local_consensus', color_discrete_map=COLOR_MAP,
                        title="Distribuzione Finale"
                    )
                    st.plotly_chart(fig_local, use_container_width=True)
                    st.dataframe(merged_local[['video_id', 'title', 'local_consensus']].head(50), use_container_width=True)

if __name__ == "__main__":
    main()