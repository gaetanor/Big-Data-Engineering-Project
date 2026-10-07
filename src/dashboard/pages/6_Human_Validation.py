import streamlit as st
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

# Setup per importare utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(page_title="Human Validation", page_icon="🧪", layout="wide")

# Percorsi ai file
DATA_LAKE = Path(__file__).resolve().parent.parent.parent.parent / "data_lake"
LLM_FILE = DATA_LAKE / "final_consensus_dataset.csv"
HUMAN_FILE = DATA_LAKE / "human_review_COMPLETED.csv"
HUMAN_SAMPLE_FILE = DATA_LAKE / "human_review.csv"
MATRIX_IMG = DATA_LAKE / "confusion_matrix.png"

ALLOWED_LABELS = ["reliable", "misleading", "false", "satire", "out-of-context"]

def main():
    st.title("Validazione Human-in-the-Loop")
    st.markdown("""
    Questa pagina descrive il rigoroso processo di validazione manuale utilizzato per misurare l'accuratezza 
    della giuria LLM rispetto al **Ground Truth** fornito da un esperto umano.
    """)

    # Creazione delle due schede
    tab1, tab2 = st.tabs(["📊 Report Statico", "🕹️ Annotazione Live"])

    # ==========================================
    # TAB 1: REPORT STATICO
    # ==========================================
    with tab1:
        # --- SPIEGAZIONE DEL PROCEDIMENTO ---
        st.header("Procedimento di Validazione")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Step 1: Campionamento")
            st.write("""
            È stato estratto un **campione stratificato** dal dataset globale. 
            Questo garantisce che le proporzioni delle etichette nel campione siano identiche a quelle del dataset originale.
            """)
        with col2:
            st.subheader("Step 2: Blind Review")
            st.write("""
            L'analista umano ha valutato i video in modalità **cieca**, senza conoscere 
            il verdetto emesso dai modelli AI, per evitare bias di conferma.
            """)
        with col3:
            st.subheader("Step 3: Confronto")
            st.write("""
            I giudizi umani sono stati confrontati con il consenso della giuria LLM per 
            calcolare l'accuratezza e identificare eventuali pattern di errore dell'AI.
            """)

        st.divider()

        # --- CARICAMENTO DATI ---
        if not HUMAN_FILE.exists():
            st.error("⚠️ Il file 'human_review_COMPLETED.csv' non è stato trovato. Compilalo tramite la tab 'Annotazione Live' oppure caricalo manualmente.")
        else:
            # Merge dei dati per il confronto
            df_llm = pd.read_csv(LLM_FILE)
            df_human = pd.read_csv(HUMAN_FILE)
            
            # Pulizia e validazione
            df_human['human_label'] = df_human['human_label'].astype(str).str.strip().str.lower()
            df_human = df_human[df_human['human_label'].isin(ALLOWED_LABELS)]
            df_comparison = pd.merge(df_human, df_llm[['video_id', 'consensus_label']], on='video_id', how='inner')

            # --- METRICHE DI PERFORMANCE ---
            st.header("2. Performance del Sistema")
            
            acc = accuracy_score(df_comparison['human_label'], df_comparison['consensus_label'])
            
            st.metric("Accuratezza Globale (LLM vs Umano)", f"{acc * 100:.2f}%")

            st.subheader("Classification Report")
            report = classification_report(
                df_comparison['human_label'], 
                df_comparison['consensus_label'], 
                output_dict=True,
                zero_division=0
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.2f}"), use_container_width=True)
            st.info("💡 **Precision**: Quanto l'AI è precisa quando dice che un video è 'False'.\n\n**Recall**: Quanti dei video realmente 'False' l'AI è riuscita a trovare.")

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader("Matrice di Confusione")
            if MATRIX_IMG.exists():
                col_sx, col_img, col_dx = st.columns([1, 4, 1])
                with col_img:
                    st.image(str(MATRIX_IMG), use_container_width=True)
            else:
                st.warning("Immagine della matrice non trovata in data_lake. Esegui 'hitL_evaluation.py' per generarla.")

            st.divider()

            # --- ANALISI DEI CASI DI DISACCORDO ---
            st.header("3. Analisi degli Errori")
            st.write("In questa tabella mostriamo i video dove l'AI ha fallito rispetto al giudizio umano.")
            
            disaccordi = df_comparison[df_comparison['human_label'] != df_comparison['consensus_label']]
            
            if not disaccordi.empty:
                st.table(disaccordi[['title', 'human_label', 'consensus_label']])
            else:
                st.success("Non ci sono disaccordi nel campione analizzato!")

    # ==========================================
    # TAB 2: ANNOTAZIONE INTERATTIVA LIVE
    # ==========================================
    with tab2:
        #st.header("Annotazione Live")
        st.subheader("Valuta il video, senza sbirciare il voto dell'AI.")

        if not HUMAN_SAMPLE_FILE.exists():
            st.warning("Esegui `hitL_sampling.py` per generare il file di campionamento prima di usare la Live Demo.")
        else:
            live_df = pd.read_csv(HUMAN_SAMPLE_FILE)
            
            # Gestione sicura dei valori nulli (NaN) in Pandas
            live_df['human_label'] = live_df['human_label'].fillna("")
            live_df['human_notes'] = live_df['human_notes'].fillna("")
            
            # Trova i video non ancora annotati
            unannotated = live_df[live_df['human_label'] == ""]
            
            # Barra di progresso
            total = len(live_df)
            done = total - len(unannotated)
            progress_ratio = done / total if total > 0 else 0
            st.progress(progress_ratio, text=f"Progresso annotazione: {done}/{total} video completati")

            if unannotated.empty:
                st.success("🎉 Hai completato l'annotazione di tutto il campione!")
                st.info("💡 **Prossimo Step**: Rinomina il file `human_review.csv` in `human_review_COMPLETED.csv`, esegui `hitL_evaluation.py` e ricarica questa pagina per vedere i risultati aggiornati nel Report Statico!")
            else:
                current_task = unannotated.iloc[0]
                idx_to_update = unannotated.index[0]

                st.subheader("Video in esame:")
                st.info(f"**Titolo:** {current_task['title']}\n\n**Piattaforma:** {current_task['platform']} | **Dominio:** {current_task['topic']}")
                
                # Se presente, mostra il link cliccabile
                if 'url' in current_task and pd.notna(current_task['url']):
                    st.write(f"[🔗 Clicca qui per aprire il video originale su {current_task['platform']}]({current_task['url']})")

                # Form per l'inserimento del giudizio e della Rationale (Stile LLM)
                with st.form("annotation_form"):
                    st.write("🎯 **Esprimi un giudizio:**")
                    
                    selected_label = st.radio("Seleziona la Categoria (Label):", ALLOWED_LABELS, horizontal=True)
                    
                    # text_area permette all'umano di scrivere paragrafi strutturati come fa l'AI
                    rationale = st.text_area(
                        "Motivazione (Rationale) - Opzionale:", 
                        placeholder="Spiega il ragionamento logico dietro la tua scelta per un confronto diretto con la rationale dell'AI..."
                    )
                    
                    submitted = st.form_submit_button("Salva Valutazione Umana", type="primary")
                    
                    if submitted:
                        # Aggiornamento DataFrame
                        live_df.at[idx_to_update, 'human_label'] = selected_label
                        # Salviamo la rationale nella colonna pre-esistente per non rompere i CSV
                        live_df.at[idx_to_update, 'human_notes'] = rationale 
                        
                        # Salvataggio su disco
                        live_df.to_csv(HUMAN_SAMPLE_FILE, index=False)
                        
                        st.success("✅ Verdetto e Rationale salvati con successo! Caricamento del prossimo video...")
                        st.rerun() # Ricarica istantaneamente la pagina per mostrare il prossimo video da annotare

if __name__ == "__main__":
    main()