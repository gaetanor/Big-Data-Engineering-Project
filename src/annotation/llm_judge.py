import asyncio
import json
import os
import logging
import cv2
import pytesseract
from collections import Counter
from pathlib import Path
from dotenv import load_dotenv
import pymongo

# --- SDK PROPRIETARI ---
from openai import AsyncOpenAI

# --- SETUP E INIZIALIZZAZIONE ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_MultiAgent_Jury")

load_dotenv()

# --- CONFIGURAZIONE MONGODB ---
MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"

# --- INIZIALIZZAZIONE CLIENT API ---
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# --- LA GIURIA ---
# Ogni entry è una tupla (provider, model_id).
# Provider supportati: "openai" | "gemini" | "openrouter"
MODELS = {
    "Judge_GPT_4o":     ("openai",      "gpt-4o"),
    "Judge_Gemini-3.1": ("openrouter",      "google/gemini-3.1-flash-lite"),
    "Judge_Llama_70B":  ("openrouter",  "meta-llama/llama-3.3-70b-instruct"),
    "Judge_DeepSeek":    ("openrouter",  "deepseek/deepseek-v4-flash"),
    "Judge_Qwen_3.5":   ("openrouter",  "qwen/qwen3-235b-a22b-2507"),
}

ALLOWED_LABELS = ["reliable", "misleading", "false", "satire", "out-of-context"]

# --- PATH DEL DATA LAKE ---
DATA_LAKE_DIR = Path("data_lake")
VIDEOS_DIR = DATA_LAKE_DIR / "videos"
ANNOTATIONS_DIR = DATA_LAKE_DIR / "annotations"

ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

def get_mongo_collection():
    mongo_client = pymongo.MongoClient(MONGO_URI)
    return mongo_client[DB_NAME][COLLECTION_NAME]

# --- FUNZIONI UTILITY PER AUDIO/TESTO ---

def is_music_or_empty(transcript: str) -> bool:
    if not transcript or not isinstance(transcript, str):
        return True
    t_clean = transcript.lower().strip()
    if len(t_clean) < 15:
        return True
    music_tags = ["[music]", "(music)", "🎵", "♪", "[musica]", "audio playback"]
    if any(tag in t_clean for tag in music_tags) and len(t_clean) < 30:
        return True
    return False

def extract_text_with_ocr(video_id: str) -> str:
    video_files = list(VIDEOS_DIR.glob(f"*{video_id}*.mp4"))
    if not video_files:
        logger.warning(f"File video per OCR non trovato: {video_id}")
        return ""
    video_path = video_files[0]
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            return ""
        frames_to_sample = [
            int(frame_count * 0.25),
            int(frame_count * 0.50),
            int(frame_count * 0.75),
        ]
        extracted_texts = []
        for f_idx in frames_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).strip()
                if text:
                    extracted_texts.append(text)
        cap.release()
        final_text = " | ".join(set(extracted_texts))
        return final_text if len(final_text) > 10 else ""
    except Exception as e:
        logger.error(f"Errore OCR sul video {video_id}: {e}")
        return ""

def build_prompt(video_document: dict, transcript_text: str, is_ocr: bool) -> str:
    testo_label = "[TESTO A SCHERMO (OCR)]" if is_ocr else "[TRASCRIZIONE AUDIO]"
    if not transcript_text:
        transcript_text = "NESSUN AUDIO O TESTO RILEVATO."

    hashtags = video_document.get("hashtags", [])
    hashtags_str = ", ".join(hashtags) if isinstance(hashtags, list) else str(hashtags)

    prompt = f"""[METADATI VIDEO]
- ID: {video_document.get('video_id', 'N/A')}
- Piattaforma: {video_document.get('platform', 'N/A')}
- Uploader: {video_document.get('uploader', 'N/A')}
- Titolo: {video_document.get('title', 'N/A')}
- Hashtag: {hashtags_str}

{testo_label}
{transcript_text}
"""
    return prompt


# --- ORCHESTRAZIONE AGENTI LLM CON ROUTING PER PROVIDER ---

async def get_model_verdict(
    model_name: str,
    model_info: tuple,
    prompt: str,
    system_instruction: str = None,
) -> dict:
    """
    Chiama il modello corretto in base al provider specificato nella tupla model_info.
    model_info = (provider, model_id)
    """
    provider, model_id = model_info

    if system_instruction is None:
        system_instruction = (
            'You are a Senior Misinformation Analyst. Analyze the content and classify it '
            'into EXACTLY ONE of these categories: "reliable", "misleading", "false", "satire", "out-of-context".\n'
            'CRITICAL: Return ONLY a valid JSON object. Do NOT wrap it in markdown. '
            'Example: {"label": "false", "rationale": "reason..."}'
        )

    try:
        raw_output = ""

        # ROUTE 1: OpenAI (GPT-4o)
        if provider == "openai":
            response = await openai_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Analyze this and output JSON:\n\n{prompt}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw_output = response.choices[0].message.content
            
        # ROUTE 2: OpenRouter (modelli vari)
        elif provider == "openrouter":
            response = await openrouter_client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": f"Analyze this and output JSON:\n\n{prompt}"},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            raw_output = response.choices[0].message.content

        else:
            raise ValueError(f"Provider sconosciuto: '{provider}'. Usa 'openai', 'gemini' o 'openrouter'.")

        # --- Parsing unificato ---
        clean_json = raw_output.strip().replace("```json", "").replace("```", "")
        verdict = json.loads(clean_json)

        if verdict.get("label") not in ALLOWED_LABELS:
            verdict["label"] = "error"

        return verdict

    except Exception as e:
        logger.error(f"Errore da {model_name} ({model_id}): {e}")
        return {"label": "error", "rationale": str(e)}


async def execute_tie_break(
    tiebreaker_name: str,
    tiebreaker_info: tuple,
    content_prompt: str,
    tied_labels: list,
) -> dict:
    logger.warning(
        f"⚖️ TIE-BREAK! Il giudice {tiebreaker_name} risolverà lo stallo "
        f"tra '{tied_labels[0]}' e '{tied_labels[1]}'."
    )
    tiebreak_instruction = (
        f'You are the Tie-Breaker Judge for a misinformation analysis panel.\n'
        f'The jury is deadlocked 2 to 2 between "{tied_labels[0]}" and "{tied_labels[1]}".\n'
        f'You MUST act as the tie-breaker. Analyze the content and STRICTLY choose EXACTLY ONE of those two tied categories.\n'
        f'Return ONLY a valid JSON object. Example: {{"label": "{tied_labels[0]}", "rationale": "I break the tie because..."}}'
    )
    return await get_model_verdict(
        f"TieBreak_{tiebreaker_name}",
        tiebreaker_info,
        content_prompt,
        tiebreak_instruction,
    )


async def process_single_video(video_document: dict, db_collection):
    video_id = video_document.get("video_id")
    if not video_id:
        return

    output_path = ANNOTATIONS_DIR / f"{video_id}_annotation.json"

    if video_document.get("llm_misinfo_consensus"):
        logger.info(f"⏭️  {video_id} già annotato in DB. Salto.")
        return

    if output_path.exists():
        logger.info(f"⏭️  {video_id} già annotato su disco. Salto.")
        return

    transcript_text = video_document.get("transcript", "")
    is_ocr_used = False

    if is_music_or_empty(transcript_text):
        logger.info(f"🔎 Nessun parlato in {video_id}. Avvio OCR...")
        ocr_text = extract_text_with_ocr(video_id)
        if ocr_text:
            transcript_text = ocr_text
            is_ocr_used = True
        else:
            transcript_text = ""

    content_prompt = build_prompt(video_document, transcript_text, is_ocr=is_ocr_used)
    logger.info(f"⚖️  Convocazione Giuria d'Elite per {video_id}...")

    # Lancia tutti i giudici in parallelo
    tasks = [
        get_model_verdict(name, model_info, content_prompt)
        for name, model_info in MODELS.items()
    ]
    results = await asyncio.gather(*tasks)
    judgments = {name: verdict for name, verdict in zip(MODELS.keys(), results)}

    votes = [v["label"] for v in judgments.values() if v["label"] != "error"]
    counts = Counter(votes)
    most_common = counts.most_common()

    consensus_label = "ambiguous"
    tiebreak_data = None

    if most_common:
        top_votes = most_common[0][1]

        if top_votes >= 3:
            # Maggioranza chiara (3+ voti uguali)
            consensus_label = most_common[0][0]

        elif top_votes == 2 and (len(most_common) == 1 or most_common[1][1] == 1):
            # Un'etichetta ha 2 voti, le altre ne hanno al più 1
            consensus_label = most_common[0][0]

        elif top_votes == 2 and most_common[1][1] == 2:
            # Pareggio 2-2: attiva il tie-break
            tied_labels = [most_common[0][0], most_common[1][0]]
            tiebreaker_name = None

            # Cerca un giudice che non sia tra i pareggiati (il suo verdetto sarà imparziale)
            for judge_name, data in judgments.items():
                if data["label"] not in tied_labels and data["label"] != "error":
                    tiebreaker_name = judge_name
                    break

            if tiebreaker_name:
                tiebreaker_info = MODELS[tiebreaker_name]
                tie_verdict = await execute_tie_break(
                    tiebreaker_name, tiebreaker_info, content_prompt, tied_labels
                )
                consensus_label = tie_verdict["label"]
                tiebreak_data = {
                    "judge": tiebreaker_name,
                    "options": tied_labels,
                    "final_decision": tie_verdict,
                }
            else:
                consensus_label = "ambiguous"

    final_report = {
        "video_id": video_id,
        "used_ocr": is_ocr_used,
        "consensus_label": consensus_label,
        "judgments": judgments,
    }

    if tiebreak_data:
        final_report["tiebreak_event"] = tiebreak_data

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4, ensure_ascii=False)

    try:
        db_collection.update_one(
            {"_id": video_document["_id"]},
            {"$set": {
                "llm_misinfo_consensus": consensus_label,
                "used_ocr_misinfo": is_ocr_used,
            }},
        )
    except Exception as e:
        logger.error(f"Errore aggiornamento DB per {video_id}: {e}")

    logger.info(f"✅ {video_id} Completato -> Consenso: {consensus_label.upper()}")


async def run_batch_annotation():
    db_collection = get_mongo_collection()

    query = {"llm_misinfo_consensus": {"$exists": False}}
    videos_to_annotate = list(db_collection.find(query))
    total_videos = len(videos_to_annotate)

    if total_videos == 0:
        logger.info("🎉 Tutti i video nel database sono già stati annotati!")
        return

    logger.info(f"📁 Trovati {total_videos} video da processare da MongoDB.")

    for i, video_doc in enumerate(videos_to_annotate, 1):
        logger.info(f"--- Processando video {i}/{total_videos} ---")
        await process_single_video(video_doc, db_collection)

        if i < total_videos:
            await asyncio.sleep(2.0)


if __name__ == "__main__":
    asyncio.run(run_batch_annotation())