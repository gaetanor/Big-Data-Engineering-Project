import os
import json
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# Assicurati di avere GROQ_API_KEY nel .env

load_dotenv()  

def generate_video_explanation(video_id: str, metadata: dict, annotations: dict) -> str:
    """
    Sintetizza solo le motivazioni dei giudici che concordano con il consenso finale.
    """
    # Inizializzazione del modello (Llama 3.1 8B è ideale per velocità e costi)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)
    
    # Recuperiamo il verdetto finale (Consensus)
    consensus = metadata.get('consensus_label')
    
    # 1. FILTRAGGIO: Selezioniamo solo le motivazioni concordanti
    judgments_text = ""
    found_matching = False
    
    for judge, verdict in annotations.get("judgments", {}).items():
        # Verifichiamo se il label del singolo LLM corrisponde al consenso finale
        if verdict.get('label') == consensus:
            judgments_text += f"\n- {judge} motiva la scelta '{consensus}' dicendo: {verdict['rationale']}"
            found_matching = True
            
    # Gestione del caso "Ambiguous" o nessun match (es. errore o pareggio totale)
    if not found_matching:
        if consensus == "ambiguous":
            return "La giuria non ha raggiunto una maggioranza. Ogni modello ha fornito un'interpretazione diversa, indicando un alto livello di ambiguità nel contenuto del video."
        return "Non è stato possibile recuperare motivazioni concordanti con il verdetto finale."

    # 2. Configurazione del Prompt per il Sintetizzatore
    system_template = """Sei un 'Explanation Agent' per una dashboard di analisi dei Big Data.
    Ti verranno fornite le motivazioni dei giudici (LLM) che hanno concordato sul verdetto finale di un video.
    
    Il tuo compito è fondere queste motivazioni in una spiegazione unica, fluida e professionale per l'utente.
    
    REGOLE:
    1. NON citare i nomi dei singoli modelli (es. Judge_Llama). Parla della 'giuria' o del 'sistema'.
    2. Sii conciso (massimo 3-4 frasi).
    3. Rispondi SEMPRE in italiano.
    4. Non aggiungere informazioni che non siano presenti nelle motivazioni fornite.
    """
    
    user_template = """
    Dati del Video:
    - Titolo: {title}
    - Verdetto di Consenso: {consensus}
    
    Motivazioni della Giuria (solo giudici concordanti):
    {judgments}
    
    Genera una sintesi discorsiva che spieghi all'utente il perché di questa classificazione.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    
    # 3. Esecuzione della Chain
    chain = prompt | llm
    
    try:
        response = chain.invoke({
            "title": metadata.get('title', 'N/A'),
            "consensus": consensus.upper() if consensus else "N/A",
            "judgments": judgments_text
        })
        return response.content
    except Exception as e:
        # Se Groq è down, restituiamo un fallback invece di far crashare la dashboard
        return f"⚠️ Errore di connessione all'Agente LLM. Impossibile generare la sintesi in questo momento. (Dettaglio: {str(e)})"