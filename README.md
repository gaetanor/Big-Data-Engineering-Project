# Progetto di Big Data Engineering - Università degli Studi di Napoli Federico II

Questo progetto implementa un'architettura Data Engineering end-to-end per raccogliere, elaborare e analizzare l'affidabilità dei contenuti video brevi (Shorts, Reels, TikTok) legati a tematiche di Finanza, Nutrizione e Politica. Il sistema sfrutta tecniche avanzate di Natural Language Processing (NLP), Computer Vision (OCR) e un'architettura **Multi-Agent LLM-as-a-Judge** per classificare la disinformazione.

---

# 🛠️ Prerequisiti di Sistema
Prima di avviare il progetto, assicurarsi di avere i seguenti requisiti installati:

* **Python**: Versione 3.10 o superiore.
* **MongoDB**: In esecuzione localmente sulla porta `27017` (con credenziali configurate) o tramite container Docker.
* **Tesseract OCR**: Richiesto per l'estrazione del testo a schermo nei video muti.
  * *macOS*: `brew install tesseract`
  * *Linux*: `sudo apt-get install tesseract-ocr`
* **FFmpeg**: Richiesto per l'estrazione audio tramite librerie Whisper.
  * *macOS*: `brew install ffmpeg`
  * *Linux*: `sudo apt-get install ffmpeg`

---

# ⚙️ Configurazione dell'Ambiente

1. **Clonare il repository e accedere alla cartella radice del progetto**

2. **Creare e attivare l'ambiente virtuale**
    python -m venv .venv
    source .venv/bin/activate  # Su Windows: .venv\Scripts\activate

3. **Installare le dipendenze**
    pip install -r requirements.txt

4. **Configurare le variabili d'ambiente**
    Creare un file .env nella directory radice e inserire le chiavi API e l'URI del database:
    MONGO_URI="mongodb://admin:password123@localhost:27017/?authSource=admin"
    OPENAI_API_KEY=""
    OPENROUTER_API_KEY=""

--

# 🚀 Pipeline di Esecuzione
Assicurarsi di trovarsi nella cartella radice del progetto e di avere l'ambiente virtuale attivato.

1. **Data Collection**

- TikTok 400 video
python -m src.collection.tiktok.main --topic finance --extra-keywords crypto trading money --target 100
python -m src.collection.tiktok.main --topic nutrition --extra-keywords diet healthy fitness --target 250
python -m src.collection.tiktok.main --topic politics --extra-keywords news election government --target 400

- Instagram 300 video
python -m src.collection.instagram.main --topic finance --extra-keywords crypto trading money --target 500
python -m src.collection.instagram.main --topic nutrition --extra-keywords diet healthy fitness --target 600
python -m src.collection.instagram.main --topic politics --extra-keywords news election government --target 700

- Youtube 300 video
python -m src.collection.youtube.main --topic finance --extra-keywords crypto trading money --target 800
python -m src.collection.youtube.main --topic nutrition --extra-keywords diet healthy fitness --target 900
python -m src.collection.youtube.main --topic politics --extra-keywords news election government --target 1000

In totale avremo:
- Finance: 300
- Nutrition: 350
- Politics: 350

2. **Data Processing**
python -m src.processing.spark_etl

3. **Annotation Task 2**
python -m src.annotation.task2_annotator

4. **LLM-as-a-Judge Task 3**
python -m src.annotation.llm_judge
python -m src.annotation.metrics

5. **Human-in-the-Loop**
python -m src.annotation.hitL_sampling 

# Valutazione Manuale:
- Avviare la dashboard, andare in Human Validation -> Tab "Annotazione Live", valutare i video finché non appare il messaggio di successo.
- Oppure aprire il file .csv e aggiungere l'etichetta nella colonna human_label. 
Infine rinominare manualmente il file nel data_lake da human_review.csv a human_review_COMPLETED.csv

python -m src.annotation.hitL_evaluation

6. **Dashboard**
streamlit run src/dashboard/Home.py