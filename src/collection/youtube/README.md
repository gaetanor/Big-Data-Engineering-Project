# Modulo YouTube Shorts Scraper

Questo modulo raccoglie automaticamente metadati di **YouTube Shorts**, li trascrive localmente con **faster-whisper** e li salva in formato JSONL, pronto per essere usato come dataset in tesi o ricerche.

---

## Indice

1. [Prerequisiti](#1-prerequisiti)
2. [Installazione](#2-installazione)
3. [Avvio rapido](#3-avvio-rapido)
4. [Opzioni da riga di comando](#4-opzioni-da-riga-di-comando)
5. [Configurazione avanzata via JSON](#5-configurazione-avanzata-via-json)
6. [Come funziona internamente](#6-come-funziona-internamente)
7. [Struttura dei dati di output](#7-struttura-dei-dati-di-output)
8. [Pipeline di trascrizione](#8-pipeline-di-trascrizione)
9. [Come riprendere una raccolta interrotta](#9-come-riprendere-una-raccolta-interrotta)
10. [Domande frequenti](#10-domande-frequenti)

---

## 1. Prerequisiti

- Python **3.10** o superiore
- Connessione a internet
- Un terminale (PowerShell, CMD, bash, ecc.)

---

## 2. Installazione

Dalla cartella radice del progetto (`Scraper/`):

```bash
# Crea e attiva l'ambiente virtuale
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Installa le dipendenze
pip install -r requirements.txt
```

> **Dipendenze principali:**
> | Pacchetto | A cosa serve |
> |-----------|--------------|
> | `yt-dlp` | Scarica metadati e audio da YouTube senza API key |
> | `faster-whisper` | Trascrizione audio locale (pipeline primaria) |
> | `tqdm` | Barra di avanzamento nel terminale |
> | `pandas` | (disponibile per analisi successive) |
> | `pyarrow` | (disponibile per salvare in Parquet) |

Per la pipeline opzionale **WhisperX** (con diarizzazione speaker):
```bash
pip install whisperx
```

---

## 3. Avvio rapido

**Sempre da dentro la cartella `youtube/`:**

```bash
cd youtube
```

Raccoglie 10.000 Shorts sul topic *cooking* con trascrizione automatica:

```bash
python main.py --topic cooking
```

Raccoglie Shorts su più topic contemporaneamente:

```bash
python main.py --topic cooking fitness gaming
```

Aggiunge parole chiave extra per avere più varietà di query:

```bash
python main.py --topic cooking --extra-keywords easy beginner tutorial
```

Imposta un target diverso (es. 2.000 video):

```bash
python main.py --topic cooking --target 2000
```

Attiva anche la trascrizione WhisperX con diarizzazione speaker:

```bash
python main.py --topic cooking --whisper --hf-token <tuo_token_huggingface>
```

---

## 4. Opzioni da riga di comando

```
python main.py [opzioni]
```

| Opzione | Tipo | Default | Descrizione |
|---------|------|---------|-------------|
| `--topic` | lista di stringhe | `cooking` | Hashtag topic da cercare (senza `#`) |
| `--extra-keywords` | lista di stringhe | *(vuoto)* | Parole chiave aggiuntive per variare le query |
| `--target` | intero | `10000` | Numero di video unici da raccogliere |
| `--results-per-query` | intero | `500` | Risultati massimi per ogni chiamata a yt-dlp |
| `--sleep` | float (secondi) | `4.0` | Pausa tra una query e la successiva |
| `--data-dir` | percorso | `data/` | Cartella dove salvare i file di output |
| `--config` | percorso file | *(nessuno)* | File JSON di configurazione alternativo |
| `--export-only` | flag | — | Salta la raccolta, esporta solo `raw.jsonl → dataset.jsonl` |
| `--stats` | flag | — | Mostra quanti record sono stati raccolti e termina |
| `--whisper` | flag | — | Attiva la pipeline WhisperX (trascrizione + diarizzazione speaker) |
| `--whisper-model` | stringa | `base` | Modello Whisper: `tiny` `base` `small` `medium` `large-v2` |
| `--whisper-device` | stringa | `cpu` | Device: `cpu` oppure `cuda` (GPU) |
| `--hf-token` | stringa | *(vuoto)* | Token HuggingFace per la diarizzazione speaker con WhisperX |
| `--verbose` | flag | — | Mostra log dettagliati e output di yt-dlp |

### Esempi pratici

```bash
# Controlla quanti video hai già raccolto senza avviare nuova raccolta:
python main.py --stats

# Esporta il dataset finale senza raccogliere altri video:
python main.py --export-only

# Raccolta con log dettagliati (utile per debug):
python main.py --topic fitness --verbose

# Usa una cartella dati personalizzata:
python main.py --topic cooking --data-dir miei_dati/cooking

# WhisperX su GPU con modello large:
python main.py --topic cooking --whisper --whisper-model large-v2 --whisper-device cuda --hf-token hf_xxx
```

---

## 5. Configurazione avanzata via JSON

Per parametrizzare la raccolta in modo riproducibile, puoi creare un file JSON. Esempio `config_cooking.json`:

```json
{
  "topics": ["cooking", "baking"],
  "extra_keywords": ["easy", "quick", "5minutes"],
  "target_count": 5000,
  "results_per_query": 300,
  "sleep_between_queries": 6.0,
  "data_dir": "data/cooking",
  "whisper_model": "small",
  "whisper_device": "cpu"
}
```

Avvio con file di configurazione:

```bash
python main.py --config config_cooking.json
```

Le opzioni CLI hanno priorità sul file JSON (utile per fare override al volo):

```bash
# Usa il JSON ma cambia solo il target:
python main.py --config config_cooking.json --target 1000
```

**Tutti i parametri disponibili nel JSON:**

| Chiave | Default | Descrizione |
|--------|---------|-------------|
| `topics` | `["cooking"]` | Lista di topic |
| `extra_keywords` | `[]` | Parole chiave extra |
| `target_count` | `10000` | Numero target di video |
| `results_per_query` | `500` | Risultati per chiamata yt-dlp |
| `max_duration` | `60` | Durata massima in secondi (Shorts sono ≤ 60s) |
| `sleep_between_queries` | `4.0` | Secondi di pausa tra query |
| `backoff_base` | `30.0` | Attesa iniziale in caso di rate limit (secondi) |
| `backoff_max_wait` | `300.0` | Attesa massima in caso di rate limit (secondi) |
| `max_retries` | `5` | Tentativi massimi prima di abbandonare una query |
| `data_dir` | `"data"` | Cartella di output |
| `transcript_langs` | `["it", "en"]` | Lingue preferite per la trascrizione |
| `use_whisper` | `false` | Attiva la pipeline WhisperX |
| `whisper_model` | `"base"` | Modello Whisper (`tiny` `base` `small` `medium` `large-v2`) |
| `whisper_device` | `"cpu"` | Device (`cpu` o `cuda`) |
| `whisper_language` | `null` | Forza una lingua specifica (es. `"it"`), `null` = auto-detect |
| `whisper_hf_token` | `""` | Token HuggingFace per diarizzazione speaker |
| `verbose` | `false` | Log dettagliati |

---

## 6. Come funziona internamente

Lo scraper è composto da 5 file Python. Di seguito spieghiamo il ruolo di ognuno e il flusso di esecuzione.

### 6.1 Architettura dei file

```
youtube/
├── main.py         → punto di ingresso, CLI, orchestrazione del loop principale
├── scraper.py      → logica di ricerca e parsing dei metadati (usa yt-dlp)
├── config.py       → configurazione (parametri con valori di default)
├── storage.py      → salvataggio su disco, deduplicazione, export finale
├── transcriber.py  → pipeline di trascrizione audio (faster-whisper + WhisperX)
└── README.md       → questa documentazione
```

### 6.2 Flusso di esecuzione passo per passo

```
main.py: legge argomenti CLI
    │
    ▼
config.py: costruisce ScraperConfig (defaults → JSON file → CLI overrides)
    │
    ▼
storage.py: DataStore carica seen_ids.txt in memoria (lista video già raccolti)
    │
    ▼
scraper.py: generate_queries() → genera tutte le combinazioni di query
    │        (es: "#cooking #shorts", "cooking shorts", "#cooking #viral", ...)
    │
    ▼  Per ogni query (in ordine casuale):
    │
    ├─► scraper.py: fetch_with_backoff(query)
    │       └─ chiama yt-dlp con "ytsearch500:query"
    │          Restituisce lista di metadati "piatti" (senza aprire ogni video)
    │
    ├─► Per ogni entry nella lista:
    │
    │   [FASE 1] is_duration_candidate(entry)
    │       └─ scarta i video più lunghi di 60 secondi
    │          (informazione già presente nei metadati piatti, zero costo)
    │
    │   [FASE 2] is_short(video_id)
    │       └─ HEAD request a https://youtube.com/shorts/{id}
    │          200 = è uno Short ✓ | 303 = non è uno Short ✗
    │
    │   [FASE 3] fetch_full_info(video_id)
    │       └─ yt-dlp legge la pagina completa dello Short
    │          per ottenere view_count, like_count, commenti, tag, ecc.
    │
    │   [FASE 4] parse_record(entry)
    │       └─ normalizza i campi in un dizionario standard
    │
    │   [FASE 5] storage.save_record(record)
    │       └─ scrive in raw.jsonl + aggiunge id in seen_ids.txt
    │
    │   [FASE 6] transcriber.save_yt_transcript(video_id)
    │       └─ scarica audio → trascrive con faster-whisper → transcripts/{id}.txt
    │
    │   [FASE 7 — opzionale] transcriber.save_whisper_transcript(video_id)
    │       └─ riusa l'audio scaricato → WhisperX + diarizzazione → transcripts_whisper/{id}.json
    │
    ▼
storage.py: export() → raw.jsonl deduplica → dataset.jsonl
```

**Perché tre fasi di verifica (1-2-3)?**
Le fasi 1, 2 e 3 sono in ordine crescente di "costo" computazionale e di traffico verso YouTube:
- La Fase 1 è gratuita (legge dati già in memoria).
- La Fase 2 costa solo una HEAD request HTTP (pochissimi byte).
- La Fase 3 apre la pagina completa del video (richiesta più pesante).

Questo ordine minimizza il numero di richieste HTTP a YouTube, riducendo il rischio di essere temporaneamente bloccati (rate limiting).

### 6.3 Gestione del rate limiting (backoff esponenziale)

Se YouTube risponde con **HTTP 429 (Too Many Requests)**, lo scraper non si ferma: aspetta un numero crescente di secondi prima di riprovare.

```
Tentativo 1 → 429 → aspetta 30s
Tentativo 2 → 429 → aspetta 60s
Tentativo 3 → 429 → aspetta 120s
...fino a backoff_max_wait (300s) o max_retries (5)
```

Inoltre, se una query restituisce meno di 10 nuovi video, il codice assume che la query sia "esaurita" e triplica il tempo di sleep prima della query successiva.

### 6.4 Generazione delle query (`scraper.py`)

Per ogni topic fornito, vengono generate automaticamente 6 varianti di query usando template predefiniti:

```python
_TEMPLATES = [
    "#{topic} #shorts",
    "{topic} shorts",
    "#{topic} #shorts #viral",
    "#{topic}shorts",
    "#{topic} short video",
    "{topic} tiktok shorts",
]
```

Con `--extra-keywords` vengono aggiunte ulteriori varianti. Per esempio con `--topic cooking --extra-keywords easy`:
```
cooking easy shorts
#cooking #easy #shorts
```

Le query vengono mescolate casualmente prima di eseguirle, così una sessione ripresa non riparte sempre dalle stesse.

### 6.5 Storage e crash-safety (`storage.py`)

I dati vengono salvati in due file separati:

- **`raw.jsonl`**: un record JSON per riga, in modalità *append*. Non viene mai riscritto durante la raccolta, solo aggiunto in fondo. In caso di crash, i dati già scritti sono al sicuro.
- **`seen_ids.txt`**: un `video_id` per riga. Viene tenuto in memoria come `set` Python per controlli O(1) istantanei (evita duplicati senza rileggere raw.jsonl).

Se `seen_ids.txt` viene perso (es. crash improvviso), il DataStore lo **ricostruisce automaticamente** leggendo `raw.jsonl` all'avvio successivo.

Il metodo `export()` legge `raw.jsonl`, deduplicazione inclusa, e produce `dataset.jsonl`, che è il file finale da usare per le analisi.

---

## 7. Struttura dei dati di output

### 7.1 Struttura delle cartelle

```
data/
├── raw.jsonl                    → tutti i record raccolti (append-only)
├── dataset.jsonl                → export finale deduplicato
├── seen_ids.txt                 → lista degli ID già raccolti
├── audio/                       → file audio scaricati (formato nativo: .m4a / .webm)
│   ├── abc123XYZ.m4a
│   └── ...
├── transcripts/                 → trascrizioni testo (faster-whisper) — sempre presenti
│   ├── abc123XYZ.txt
│   └── ...
└── transcripts_whisper/         → trascrizioni WhisperX con diarizzazione — solo se --whisper
    ├── abc123XYZ.json
    └── ...
```

### 7.2 Campi di `dataset.jsonl`

Ogni riga è un oggetto JSON con i seguenti campi:

| Campo | Tipo | Descrizione |
|-------|------|-------------|
| `video_id` | `str` | ID univoco del video (es. `"dQw4w9WgXcQ"`) |
| `url` | `str` | URL classico `youtube.com/watch?v=...` |
| `shorts_url` | `str` | URL in formato Shorts `youtube.com/shorts/...` |
| `title` | `str` | Titolo del video |
| `description` | `str` | Descrizione (troncata a 5.000 caratteri) |
| `duration` | `int` | Durata in secondi |
| `view_count` | `int` | Visualizzazioni al momento della raccolta |
| `like_count` | `int` | Like (può essere `null` se nascosti) |
| `comment_count` | `int` | Numero di commenti |
| `uploader` | `str` | Nome del canale |
| `channel_id` | `str` | ID univoco del canale |
| `channel_url` | `str` | URL del canale |
| `upload_date` | `str` | Data di caricamento (`"YYYYMMDD"`) |
| `language` | `str` | Lingua rilevata da YouTube |
| `tags` | `list[str]` | Tag del video |
| `categories` | `list[str]` | Categorie YouTube |
| `hashtags` | `list[str]` | Hashtag estratti dalla descrizione |
| `thumbnail_url` | `str` | URL della miniatura |
| `comments` | `list` | Lista commenti (vuota per default) |
| `topic` | `str` | Topic che ha generato questa raccolta |
| `platform` | `str` | Sempre `"youtube_shorts"` |
| `search_query` | `str` | Query esatta che ha trovato questo video |
| `collected_at` | `str` | Timestamp ISO 8601 della raccolta (UTC) |

### 7.3 Esempio di un record

```json
{
  "video_id": "abc123XYZ",
  "url": "https://www.youtube.com/watch?v=abc123XYZ",
  "shorts_url": "https://www.youtube.com/shorts/abc123XYZ",
  "title": "Easy 5-Minute Pasta Recipe 🍝",
  "description": "Super quick and easy pasta! #cooking #shorts #easy",
  "duration": 42,
  "view_count": 1500000,
  "like_count": 85000,
  "comment_count": 1200,
  "uploader": "QuickCooks",
  "channel_id": "UCxyz...",
  "channel_url": "https://www.youtube.com/channel/UCxyz...",
  "upload_date": "20240315",
  "language": "en",
  "tags": ["cooking", "pasta", "easy recipe"],
  "categories": ["Howto & Style"],
  "hashtags": ["#cooking", "#shorts", "#easy"],
  "thumbnail_url": "https://i.ytimg.com/vi/abc123XYZ/hqdefault.jpg",
  "comments": [],
  "topic": "cooking",
  "platform": "youtube_shorts",
  "search_query": "#cooking #shorts",
  "collected_at": "2024-03-18T10:30:00+00:00"
}
```

---

## 8. Pipeline di trascrizione

### 8.1 Pipeline primaria — faster-whisper (sempre attiva)

Per ogni video raccolto, il modulo scarica l'audio nel formato nativo (`.m4a` o `.webm`) e lo trascrive localmente con **faster-whisper**. Non dipende dalla disponibilità dei sottotitoli su YouTube.

Il flusso è:
1. Download audio in `data/audio/{video_id}.m4a` (saltato se già presente)
2. Trascrizione con faster-whisper → testo pulito
3. Salvataggio in `data/transcripts/{video_id}.txt`

Il file `.txt` viene **sempre** creato: se la trascrizione fallisce o l'audio è vuoto, viene scritto un file vuoto, in modo che una raccolta ripresa non ritenti inutilmente.

**Parametri configurabili:**

| Parametro | Default | Descrizione |
|-----------|---------|-------------|
| `whisper_model` | `base` | Dimensione del modello. Modelli più grandi = più accurati ma più lenti |
| `whisper_device` | `cpu` | Usa `cuda` se hai una GPU NVIDIA compatibile |
| `whisper_language` | auto | Forza una lingua specifica (es. `"it"`); `null` = rilevamento automatico |

**Trade-off dimensione modello:**

| Modello | VRAM / RAM | Velocità | Accuratezza |
|---------|-----------|----------|-------------|
| `tiny` | ~1 GB | ★★★★★ | ★★☆☆☆ |
| `base` | ~1 GB | ★★★★☆ | ★★★☆☆ |
| `small` | ~2 GB | ★★★☆☆ | ★★★★☆ |
| `medium` | ~5 GB | ★★☆☆☆ | ★★★★★ |
| `large-v2` | ~10 GB | ★☆☆☆☆ | ★★★★★ |

### 8.2 Pipeline secondaria — WhisperX con diarizzazione (opt-in)

Attivata con il flag `--whisper`. Oltre alla trascrizione fornisce:
- **Allineamento word-level**: timestamp preciso per ogni parola
- **Diarizzazione speaker**: riconosce e identifica chi sta parlando (richiede token HuggingFace)

Il risultato viene salvato come JSON strutturato in `data/transcripts_whisper/{video_id}.json` con questa forma:

```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": "Ciao a tutti!",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Ciao", "start": 0.0, "end": 0.5, "score": 0.99},
        ...
      ]
    }
  ],
  "language": "it"
}
```

> **Nota:** Se non viene fornito `--hf-token`, la diarizzazione speaker viene saltata e i segmenti non avranno il campo `speaker`. La trascrizione e l'allineamento funzionano comunque senza token.

Per ottenere un token HuggingFace gratuito:
1. Registrati su [huggingface.co](https://huggingface.co)
2. Accetta i termini di [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization)
3. Genera un token in *Settings → Access Tokens*

### 8.3 Riuso dell'audio

Entrambe le pipeline usano lo stesso file audio in `data/audio/`. Se un video è già stato trascritto con faster-whisper e poi si ri-esegue con `--whisper`, l'audio **non viene riscaricato**.

---

## 9. Come riprendere una raccolta interrotta

Se la raccolta viene interrotta (Ctrl+C, chiusura del terminale, blackout, ecc.), **non si perde nulla**. I dati già raccolti sono in `raw.jsonl`, `seen_ids.txt`, `audio/` e `transcripts/`.

Per riprendere da dove si era rimasti, basta rieseguire **esattamente lo stesso comando**:

```bash
python main.py --topic cooking
```

Lo scraper rileverà automaticamente i video già presenti e continuerà dalla query successiva. Allo stesso modo, i file audio e le trascrizioni già esistenti vengono rilevati e saltati.

> **Nota:** Le query vengono rimescolate casualmente ad ogni avvio, quindi potrebbe raccogliere qualche video dallo stesso "blocco" prima di continuare. Questo è intenzionale per garantire varietà anche nelle sessioni riprese.

---

## 10. Domande frequenti

**Q: Ho bisogno di una API key di YouTube?**
No. Il modulo usa `yt-dlp`, che accede a YouTube come farebbe un browser normale, senza richiedere credenziali.

**Q: Quanti video posso raccogliere prima di essere bloccato?**
Dipende dalla connessione e dall'orario. Il meccanismo di backoff gestisce automaticamente i rate limit. Per raccolte molto grandi (>50.000 video), si consiglia di:
- Aumentare `--sleep` (es. `--sleep 8`)
- Suddividere la raccolta in sessioni (es. 10.000 video al giorno)
- Usare una connessione con IP diversi se possibile

**Q: Perché `like_count` è `null` per alcuni video?**
YouTube nasconde i like su alcuni canali. In quel caso yt-dlp non può recuperarli e il campo viene salvato come `null`.

**Q: Gli audio vengono conservati o cancellati?**
Vengono conservati in `data/audio/`. Questo permette di ritrascrire con un modello diverso o attivare WhisperX in seguito senza riscaricari. Se vuoi liberare spazio puoi eliminarli manualmente dopo la raccolta.

**Q: Posso usare faster-whisper senza GPU?**
Sì. Il default è `cpu`. La trascrizione è più lenta ma funziona su qualsiasi macchina. Per Shorts (video ≤ 60s) il tempo di trascrizione su CPU con `base` è in genere 5–15 secondi per video.

**Q: Come leggo il dataset in Python?**
```python
import json

records = []
with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Totale record: {len(records)}")
print(records[0])  # Primo record
```

Oppure con pandas:
```python
import pandas as pd

df = pd.read_json("data/dataset.jsonl", lines=True)
print(df.shape)
print(df.columns.tolist())
```

Per leggere le trascrizioni:
```python
from pathlib import Path

transcripts = {}
for p in Path("data/transcripts").glob("*.txt"):
    transcripts[p.stem] = p.read_text(encoding="utf-8")

# Unisci al dataframe
df["transcript"] = df["video_id"].map(transcripts)
```

**Q: Posso raccogliere video normali (non Shorts)?**
No. Il modulo è progettato specificamente per YouTube Shorts: la verifica in Fase 2 (`is_short()`) filtra automaticamente tutti i video non-Shorts.

**Q: Cosa succede se un video viene eliminato dopo la raccolta?**
Il record rimane nel dataset con tutti i metadati raccolti al momento dello scraping. I dati non vengono aggiornati retroattivamente.