from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from langdetect import detect, LangDetectException

import requests
import yt_dlp
import yt_dlp.utils
import re

from .config import ScraperConfig

logger = logging.getLogger("shorts_scraper")

# yt-dlp options: extract_flat skips fetching individual video pages,
# making each search batch much faster and lighter on YouTube's servers.
_YDL_OPTS: Dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "extract_flat": "in_playlist",
    "skip_download": True,
    "ignoreerrors": True,
    "http_headers": {"Accept-Language": "en-US,en;q=0.9"} 
}


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_TEMPLATES = [
    "#{topic} #shorts",
    "{topic} shorts",
    "#{topic} #shorts #viral",
    "#{topic}shorts",
    "#{topic} short video",
    "{topic} tiktok shorts",
]


def generate_queries(
    topics: List[str], extra_keywords: List[str]
) -> List[Tuple[str, str]]:
    """
    Genera query ibride: alcune con Exact Match per l'alta precisione,
    altre aperte per massimizzare il volume.
    """
    queries: List[Tuple[str, str]] = []
    
    for raw_topic in topics:
        topic = raw_topic.strip("#").strip()
        
        # Query 1: Estremamente stretta (Ottima per iniziare)
        queries.append((f'"{topic}" #shorts', raw_topic))
        
        for kw in extra_keywords:
            kw = kw.strip()
            
            # Query 2: Exact match sulla keyword
            queries.append((f'{topic} "{kw}" #shorts', raw_topic))
            
            # Query 3: Completamente aperta (Senza virgolette, massima resa)
            queries.append((f'{topic} {kw} #shorts', raw_topic))
            
            # Query 4: Intersezione Hashtag pura
            queries.append((f'#{topic} #{kw} #shorts', raw_topic))
            
    return queries

# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class YtDlpScraper:
    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        opts = dict(_YDL_OPTS)
        if config.verbose:
            opts["quiet"] = False
            opts["no_warnings"] = False
        self._opts = opts
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        })

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def fetch_with_backoff(self, query: str) -> List[Dict[str, Any]]:
        """
        Fetch a search batch with exponential backoff on HTTP 429.
        Returns a (possibly empty) list of raw entry dicts.
        """
        n = self.config.results_per_query
        wait = self.config.backoff_base

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._fetch_batch(query, n)
            except yt_dlp.utils.DownloadError as exc:
                msg = str(exc)
                if "429" in msg or "Too Many Requests" in msg:
                    logger.warning(
                        "HTTP 429 on attempt %d/%d. Sleeping %.0fs …",
                        attempt, self.config.max_retries, wait,
                    )
                    time.sleep(wait)
                    wait = min(wait * 2, self.config.backoff_max_wait)
                else:
                    logger.error("DownloadError (non-429): %s", msg)
                    return []
            except Exception as exc:
                logger.error("Unexpected error on attempt %d: %s", attempt, exc)
                time.sleep(min(wait, self.config.backoff_max_wait))
                wait *= 2

        logger.error("All retries exhausted for query: %s", query)
        return []

    def is_duration_candidate(self, entry: Dict[str, Any]) -> bool:
        """Phase 1: check duration."""
        duration = entry.get("duration")
        try:
            duration = int(duration)
        except (TypeError, ValueError):
            return False
        return duration <= self.config.max_duration

    def is_viral_candidate(self, entry: Dict[str, Any]) -> bool:
        """Filtro Veloce: controlla le views direttamente nei risultati di ricerca."""
        views = entry.get("view_count")
        if views is None:
            return True # Se yt-dlp non trova le views nella ricerca veloce, lo facciamo passare sulla fiducia
        try:
            return int(views) >= 1000
        except (ValueError, TypeError):
            return False

    def is_language_valid(self, entry: Dict[str, Any]) -> bool:
        """
        Filtro Lingua Tollerante: Scarta i video palesemente fuori target 
        (alfabeti non latini o lingue blacklistate) per evitare di bloccare il download.
        """
        title = entry.get("title", "")
        if not title:
            return True
            
        # 1. SCUDO ALFABETICO (Ultra-veloce)
        # Se il titolo contiene caratteri Cirillici, Arabi, Hindi o Asiatici, scarta subito.
        if re.search(r'[\u0400-\u04FF\u0600-\u06FF\u0900-\u097F\u4E00-\u9FFF]', title):
            logger.debug(f"⏭️ Scartato (Alfabeto non latino): {title[:40]}")
            return False

        # 2. RILEVAMENTO LINGUA (Solo se c'è abbastanza testo pulito)
        # Rimuoviamo gli hashtag per non confondere l'algoritmo
        clean_title = re.sub(r'#\w+', '', title).strip()
        
        if len(clean_title) > 15:
            try:
                lang = detect(clean_title)
                # Invece di richiedere 'it' o 'en' (che fallisce spesso),
                # SCARTIAMO solo le lingue che sicuramente NON vogliamo.
                blacklist = ['ru', 'ar', 'hi', 'ja', 'ko', 'zh-cn', 'tr', 'pl']
                if lang in blacklist:
                    logger.debug(f"⏭️ Scartato (Lingua in Blacklist '{lang}'): {title[:40]}")
                    return False
            except LangDetectException:
                pass # Se non capisce la lingua (es. solo emoji), lo facciamo passare
                
        return True

    def is_short(self, video_id: str) -> bool:
        """
        Verifica se un video è veramente uno YouTube Short.

        Effettua una HEAD request all'URL https://www.youtube.com/shorts/{video_id}
        - Status 200 → è uno short ✓
        - Status 303 → non è uno short (redirect) ✗

        Args:
            video_id: ID del video YouTube

        Returns:
            True se è uno short, False altrimenti
        """
        #url = f"https://www.youtube.com/shorts/{video_id}"
        '''try:
            response = requests.head(url, allow_redirects=False, timeout=5)
            is_short_flag = response.status_code == 200

            if is_short_flag:
                logger.debug(f"✓ Short confermato: {video_id}")
            else:
                logger.debug(f"✗ Non-short rilevato: {video_id} (status: {response.status_code})")

            return is_short_flag
        except requests.RequestException as e:
            logger.warning(f"Errore durante la verifica di {video_id}: {e}")
            # In caso di errore, assumiamo che sia uno short (non blocchiamo)
            return True'''
        # Per evitare problemi di rate-limiting o blocchi IP, disabilitiamo la verifica HEAD e assumiamo che i video passati dai filtri precedenti siano Shorts.
        return True

    def fetch_full_info(self, video_id: str) -> Dict[str, Any]:
        """Fetch complete metadata for a confirmed Short, con limite stringente sui commenti."""
        opts = {
            **self._opts, 
            "extract_flat": False, 
            "getcomments": True,
            # IL TRUCCO: Diciamo a yt-dlp di smettere di scaricare dopo 50 commenti,
            # ordinandoli per "Top" (più rilevanti) invece che cronologicamente.
            "extractor_args": {"youtube": ["max-comments=50", "comment_sort=top"]}
        }
        
        url = f"https://www.youtube.com/shorts/{video_id}"
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=False)
            return info or {}
        except Exception as exc:
            logger.warning("fetch_full_info failed for %s: %s", video_id, exc)
            return {}

    def parse_record(
        self, entry: Dict[str, Any], search_query: str, topic: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Map a raw yt-dlp entry dict to a normalized record dict.
        Returns None if the entry lacks a video ID.
        """
        video_id = entry.get("id") or entry.get("videoId")
        if not video_id:
            return None

        url = entry.get("webpage_url") or entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        shorts_url = f"https://www.youtube.com/shorts/{video_id}"

        duration = entry.get("duration")
        try:
            duration = int(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None

        # Casting esplicito e robusto per l'Analisi dell'Engagement
        view_count = self._safe_to_int(entry.get("view_count"))
        like_count = self._safe_to_int(entry.get("like_count"))
        comment_count = self._safe_to_int(entry.get("comment_count"))
        
        description = entry.get("description") or ""
        if len(description) > 5000:
            description = description[:5000]

        # -- INIEZIONE: Estrazione e formattazione dei commenti --
        comments_list = []
        raw_comments = entry.get("comments") or []
        for c in raw_comments[:50]:  # Limitiamo a 50 commenti per evitare JSON enormi
            comments_list.append({
                "comment_id": c.get("id", ""),
                "text": c.get("text", ""),
                "author": c.get("author", ""),
                "author_id": c.get("author_id", ""),
                "like_count": c.get("like_count", 0),
                "created_at": str(c.get("timestamp", "")) 
            })
        # --------------------------------------------------------

        return {
            "video_id": video_id,
            "url": url,
            "shorts_url": shorts_url,
            "title": entry.get("title") or "",
            "description": description,
            "duration": duration,
            "view_count": view_count,
            "like_count": like_count,
            "comment_count": comment_count,
            "uploader": entry.get("uploader") or entry.get("channel") or "",
            "channel_id": entry.get("channel_id") or entry.get("uploader_id") or "",
            "channel_url": entry.get("channel_url") or entry.get("uploader_url") or "",
            "upload_date": entry.get("upload_date") or "",
            "language": entry.get("language") or "",
            "tags": entry.get("tags") or [],
            "categories": entry.get("categories") or [],
            "hashtags": self._extract_hashtags(description),
            "thumbnail_url": entry.get("thumbnail") or "",
            "comments": comments_list,
            "topic": topic,
            "platform": "youtube_shorts",
            "search_query": search_query,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        """Extract all #hashtags from a text string."""
        import re
        return re.findall(r"#\w+", text)

    def _fetch_batch(self, query: str, n: int) -> List[Dict[str, Any]]:
        search_url = f"ytsearch{n}:{query}"
        with yt_dlp.YoutubeDL(self._opts) as ydl:
            info = ydl.extract_info(search_url, download=False)
        if not info:
            return []
        entries = info.get("entries") or []
        return [e for e in entries if e is not None]
    
    @staticmethod
    def _safe_to_int(value: Any) -> Optional[int]:
        """Converte in modo sicuro un valore metrico in un numero intero."""
        if value is None:
            return None
        
        # Se è già un intero, lo restituisce
        if isinstance(value, int):
            return value
            
        # Se è una stringa, prova a pulirla e convertirla
        if isinstance(value, str):
            # Rimuove virgole o punti usati come separatori delle migliaia (es. "1,500" -> "1500")
            cleaned_str = value.replace(",", "").replace(".", "")
            
            try:
                return int(cleaned_str)
            except ValueError:
                return None
                
        # Fallback finale
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
