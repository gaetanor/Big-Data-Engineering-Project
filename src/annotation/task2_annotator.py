import os
import pymongo
import json
from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm
import time
import cv2
import pytesseract
from pathlib import Path

# Carica le variabili d'ambiente
load_dotenv()

MONGO_URI = "mongodb://admin:password123@localhost:27017/?authSource=admin"
DB_NAME = "short_videos_db"
COLLECTION_NAME = "videos"
MODEL_NAME = "llama-3.1-8b-instant"

DATA_LAKE_DIR = Path("data_lake")
VIDEOS_DIR = DATA_LAKE_DIR / "videos"
TAXONOMY_FILE = DATA_LAKE_DIR / "dynamic_taxonomy.json"

# --- FUNZIONE DI CARICAMENTO TASSONOMIA ---
def load_dynamic_taxonomy():
    """Carica la tassonomia generata dall'LLM dal file JSON."""
    if not TAXONOMY_FILE.exists():
        print(f"⚠️ Errore: File {TAXONOMY_FILE} non trovato.")
        return {} 
    with open(TAXONOMY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# INIZIALIZZAZIONE GLOBALE
DYNAMIC_TAXONOMY = load_dynamic_taxonomy()

def get_mongo_collection():
    client = pymongo.MongoClient(MONGO_URI)
    return client[DB_NAME][COLLECTION_NAME]

def is_music_or_empty(transcript):
    if not transcript or not isinstance(transcript, str): return True
    t_clean = transcript.lower().strip()
    if len(t_clean) < 15: return True
    music_tags = ["[music]", "(music)", "🎵", "♪", "[musica]", "audio playback"]
    if any(tag in t_clean for tag in music_tags) and len(t_clean) < 30: return True
    return False

def extract_text_with_ocr(video_id):
    video_path = VIDEOS_DIR / f"{video_id}.mp4"
    if not video_path.exists(): return ""
    try:
        cap = cv2.VideoCapture(str(video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0: return ""
        
        frames_to_sample = [int(frame_count * 0.25), int(frame_count * 0.50), int(frame_count * 0.75)]
        extracted_texts = []
        for f_idx in frames_to_sample:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).strip()
                if text: extracted_texts.append(text)
        cap.release()
        
        final_text = " | ".join(set(extracted_texts))
        return final_text if len(final_text) > 10 else ""
    except Exception as e:
        print(f"Errore OCR sul video {video_id}: {e}")
        return ""

def generate_topic_with_llm(client, transcript, macro_topic, metadata, is_ocr=False):
    """
    Ora passiamo all'LLM anche i metadati (Titolo e Hashtag). 
    In questo modo, anche se audio e OCR sono vuoti, l'LLM capirà il contesto!
    """
    allowed_categories = DYNAMIC_TAXONOMY.get(macro_topic.lower(), ["General Info", "Entertainment", "News"])
    categories_str = "\n".join([f"- {cat}" for cat in allowed_categories])
    
    system_prompt = f"""
    You are an expert Data Analyst. Your task is to classify a short-form video into a specific category.
    
    CRITICAL INSTRUCTIONS:
    The video belongs to the macro-topic: '{macro_topic}'.
    You MUST classify the video into EXACTLY ONE of the following predefined categories:
    
    {categories_str}
    
    RULES:
    1. Do NOT invent new categories. You must output the exact string of one of the categories above.
    2. Analyze the Title, Hashtags, and Transcript/OCR (if available) to determine the best fit.
    3. Output ONLY the category name. No conversational text.
    """
    
    # Costruiamo un Payload ricco!
    user_content = f"Title: {metadata.get('title', 'N/A')}\n"
    user_content += f"Hashtags: {', '.join(metadata.get('hashtags', []))}\n"
    
    if is_ocr and transcript:
        user_content += f"On-screen text (OCR): {transcript}"
    elif transcript:
        user_content += f"Transcript: {transcript}"
    else:
        user_content += "Transcript/OCR: [EMPTY - Rely on Title and Hashtags]"
    
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=15
        )
        return chat_completion.choices[0].message.content.strip().strip("'\"")
    except Exception as e:
        return None

def run_batch_annotation():
    print("🚀 Avvio Pipeline di Annotazione...")
    db_collection = get_mongo_collection()
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    query = {"llm_topic_groq": {"$exists": False}}
    videos_to_annotate = list(db_collection.find(query))
    total_videos = len(videos_to_annotate)
    
    if total_videos == 0: 
        print("Tutti i video sono già stati annotati!")
        return
        
    print(f"📊 Trovati {total_videos} video da annotare.")
    
    for video in tqdm(videos_to_annotate, desc="Elaborazione (LLM/OCR)"):
        transcript = video.get("transcript", "")
        video_id = video.get("video_id", "")
        macro_topic = video.get("topic", "unknown")
        
        # Prepariamo i metadati da passare alla funzione
        metadata = {
            "title": video.get("title", ""),
            "hashtags": video.get("hashtags", [])
        }
        
        if is_music_or_empty(transcript):
            # IL VIDEO E' MUTO: Attiviamo l'OCR!
            ocr_text = extract_text_with_ocr(video_id)
            # Anche se l'OCR fallisce (ocr_text è vuoto), invochiamo comunque l'LLM!
            # L'LLM userà il Titolo e gli Hashtag per classificare il video.
            generated_topic = generate_topic_with_llm(groq_client, ocr_text, macro_topic, metadata, is_ocr=True)
        else:
            # Flusso normale con la voce (ASR)
            generated_topic = generate_topic_with_llm(groq_client, transcript, macro_topic, metadata)
            
        time.sleep(0.5) 
        
        if generated_topic:
            # 1. Recuperiamo la lista delle categorie valide per questo macro-topic
            allowed_list = DYNAMIC_TAXONOMY.get(macro_topic.lower(), [])
            
            matched_category = None
            
            # 2. Pulizia
            clean_generated = generated_topic.lower().strip('. \n\t-"\'')
                
            # 3. Match Case-Insensitive
            for allowed_cat in allowed_list:
                clean_allowed = allowed_cat.lower().strip('. \n\t-"\'')
                if clean_generated == clean_allowed or clean_generated in clean_allowed:
                    matched_category = allowed_cat
                    break
            
            if not matched_category:
                print(f"⚠️ Mismatch! LLM ha generato: '{generated_topic}' che non è in {allowed_list}")
                matched_category = "Other"
                
            db_collection.update_one(
                {"_id": video["_id"]},
                {"$set": {"llm_topic_groq": matched_category}}
            )

if __name__ == "__main__":
    run_batch_annotation()