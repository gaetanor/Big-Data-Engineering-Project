import pathlib
import shutil
import logging
from .config import ScraperConfig

logger = logging.getLogger("instagram_transcriber")

class TranscriptManager:
    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.data_dir = pathlib.Path(config.data_dir)
        self.videos_dir = self.data_dir / "videos"
        self.audio_dir = self.data_dir / "audio"
        self.transcripts_dir = self.data_dir / "transcripts"

        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self._fw_model = None

    def save_transcript(self, video_id: str, url: str) -> bool:
        out_path = self.transcripts_dir / f"{video_id}.txt"
        if out_path.exists(): return True

        audio_path = self._download_media(video_id, url)
        if not audio_path: return False

        try:
            from faster_whisper import WhisperModel
            if not self._fw_model:
                self._fw_model = WhisperModel("base", device="cpu", compute_type="int8")
            
            segments, _ = self._fw_model.transcribe(audio_path)
            text = " ".join(s.text.strip() for s in segments)
            out_path.write_text(text, encoding="utf-8")
            return bool(text)
        except Exception as e:
            logger.error(f"Errore trascrizione {video_id}: {e}")
            return False

    def _download_media(self, video_id: str, url: str) -> str:
        existing = list(self.audio_dir.glob(f"{video_id}.*"))
        if existing: return str(existing[0])

        import yt_dlp
        cookies_path = self.data_dir / "cookies.txt"
        
        opts = {
            "format": "b[ext=mp4]/best",
            "outtmpl": str(self.videos_dir / f"{video_id}.%(ext)s"),
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
                "nopostoverwrites": False, 
            }],
            "keepvideo": True,
            "cookiefile": str(cookies_path),
            "quiet": True
        }
        
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
                
            origin = self.videos_dir / f"{video_id}.mp3"
            dest = self.audio_dir / f"{video_id}.mp3"
            if origin.exists():
                shutil.move(str(origin), str(dest))
                return str(dest)
        except Exception as e:
            logger.error(f"Download fallito {video_id}: {e}")
        return None