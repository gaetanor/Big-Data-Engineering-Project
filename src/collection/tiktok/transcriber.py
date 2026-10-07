from __future__ import annotations

"""
TranscriptManager
-----------------
Handles two transcript pipelines:

1. faster-whisper (always active)
   Downloads audio → plain transcription → transcripts/{video_id}.txt
   Works on any TikTok video, independent of caption availability.

2. WhisperX with diarization (opt-in via config.use_whisper)
   Downloads audio → WhisperX transcription + alignment + speaker diarization
   → transcripts_whisper/{video_id}.json
"""

import json
import logging
import pathlib
import shutil

from typing import Any, Dict, Optional, Tuple


def _find_ffmpeg_dir() -> Optional[str]:
    """Return the directory containing ffmpeg.exe, or None if already in PATH."""
    if shutil.which("ffmpeg"):
        return None  # Already in PATH, yt-dlp will find it automatically
    # Winget installs ffmpeg under AppData — search there
    winget_base = pathlib.Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for candidate in sorted(winget_base.glob("Gyan.FFmpeg_*/ffmpeg-*/bin"), reverse=True):
        if (candidate / "ffmpeg.exe").exists():
            return str(candidate)
    return None

from .config import ScraperConfig

logger = logging.getLogger("tiktok_transcriber")


# ---------------------------------------------------------------------------
# TranscriptManager
# ---------------------------------------------------------------------------

class TranscriptManager:
    """
    Usage::

        mgr = TranscriptManager(config)
        mgr.save_transcript(video_id, video_url)        # always
        if config.use_whisper:
            mgr.save_whisper_transcript(video_id, video_url)   # opt-in

    TikTok video URLs embed the author's username, so the full URL must be
    passed explicitly (unlike YouTube where it can be rebuilt from the ID).
    """

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self.data_dir = pathlib.Path(config.data_dir)
        self.transcripts_dir = self.data_dir / "transcripts"
        self.transcripts_whisper_dir = self.data_dir / "transcripts_whisper"
        self.audio_dir = self.data_dir / "audio"
        self.videos_dir = self.data_dir / "videos"
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        if config.use_whisper:
            self.transcripts_whisper_dir.mkdir(parents=True, exist_ok=True)

        self._fw_model = None                               # faster-whisper, lazy-loaded
        self._whisperx_model = None                         # WhisperX model, lazy-loaded
        self._align_models: Dict[str, Tuple[Any, Any]] = {} # lang → (model_a, metadata)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_transcript(self, video_id: str, video_url: str) -> bool:
        """
        Download audio and transcribe with faster-whisper.
        Saves plain text to transcripts/{video_id}.txt.
        Returns True if a non-empty transcript was saved.
        """
        out_path = self.transcripts_dir / f"{video_id}.txt"
        if out_path.exists():
            return out_path.stat().st_size > 0

        audio_path = self._ensure_audio(video_id, video_url)
        if not audio_path:
            out_path.write_text("", encoding="utf-8")
            print(f"[TRANSCRIPT] {video_id}: download audio fallito", flush=True)
            return False

        print(f"[TRANSCRIPT] {video_id}: trascrizione in corso …", flush=True)
        text = self._transcribe(audio_path)
        out_path.write_text(text, encoding="utf-8")

        if text:
            print(f"[TRANSCRIPT] {video_id}: OK — {len(text)} caratteri salvati", flush=True)
            logger.debug("transcript saved: %s (%d chars)", video_id, len(text))
        else:
            print(f"[TRANSCRIPT] {video_id}: nessun testo estratto", flush=True)
            logger.debug("transcript: empty result for %s", video_id)
        return bool(text)

    def save_whisper_transcript(self, video_id: str, video_url: str) -> bool:
        """
        Download audio and run the full WhisperX pipeline
        (transcription + alignment + optional speaker diarization).
        Saves JSON to transcripts_whisper/{video_id}.json.
        Returns True if successful.
        """
        out_path = self.transcripts_whisper_dir / f"{video_id}.json"
        if out_path.exists():
            return True

        audio_path = self._ensure_audio(video_id, video_url)
        if not audio_path:
            logger.warning("whisper: audio download failed for %s", video_id)
            return False

        result = self._run_whisperx(audio_path)
        if not result:
            return False

        out_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("whisper transcript saved: %s", video_id)
        return True

    # ------------------------------------------------------------------
    # Audio download
    # ------------------------------------------------------------------

    def _ensure_audio(self, video_id: str, url: str) -> Optional[str]:
        """
        Return path to the audio file for *video_id*, downloading it to
        audio_dir if not already present.
        """
        existing = list(self.audio_dir.glob(f"{video_id}.*"))
        if existing:
            return str(existing[0])

        print(f"[TRANSCRIPT] {video_id}: download video & extraction audio …", flush=True)
        try:
            import yt_dlp  
            import shutil

            cookies_path = self.data_dir / "cookies.txt"
            
            # INIEZIONE NOSTRA: Configuriamo yt-dlp per scaricare mp4 e mp3
            opts = {
                "format": "b[ext=mp4]/best",
                "outtmpl": str(self.videos_dir / f"{video_id}.%(ext)s"),
                "postprocessors": [{
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                    "nopostoverwrites": False, 
                }],
                "keepvideo": True, # Mantiene l'mp4!
                "quiet": True,
                "no_warnings": True,
            }
            
            if cookies_path.exists():
                opts["cookiefile"] = str(cookies_path)

            ffmpeg_dir = _find_ffmpeg_dir()
            if ffmpeg_dir:
                opts["ffmpeg_location"] = ffmpeg_dir
                
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])

            # Spostiamo l'audio dalla cartella videos/ a audio/
            audio_origin = self.videos_dir / f"{video_id}.mp3"
            audio_dest = self.audio_dir / f"{video_id}.mp3"
            
            if audio_origin.exists():
                shutil.move(str(audio_origin), str(audio_dest))
                return str(audio_dest)

            return None

        except Exception as exc:
            logger.warning("audio download error for %s: %s", video_id, exc)
            return None

    # ------------------------------------------------------------------
    # faster-whisper (primary transcription)
    # ------------------------------------------------------------------

    def _transcribe(self, audio_path: str) -> str:
        """Plain transcription via faster-whisper, no diarization."""
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError:
            logger.error(
                "faster-whisper not installed. Run: pip install faster-whisper"
            )
            return ""

        try:
            device = self.config.whisper_device
            compute_type = "int8" if device == "cpu" else "float16"

            if self._fw_model is None:
                logger.info(
                    "Loading faster-whisper model '%s' on %s …",
                    self.config.whisper_model, device,
                )
                self._fw_model = WhisperModel(
                    self.config.whisper_model,
                    device=device,
                    compute_type=compute_type,
                )

            lang = self.config.whisper_language or None
            segments, _ = self._fw_model.transcribe(audio_path, language=lang)
            return " ".join(seg.text.strip() for seg in segments)

        except Exception as exc:
            logger.warning("transcription error on %s: %s", audio_path, exc)
            return ""

    # ------------------------------------------------------------------
    # WhisperX (secondary pipeline with diarization)
    # ------------------------------------------------------------------

    def _run_whisperx(self, audio_path: str) -> Optional[Dict[str, Any]]:
        """
        Run WhisperX pipeline:
          1. Transcribe with Whisper
          2. Align word-level timestamps
          3. Diarize speakers (if HF token is configured)
        Returns result dict for JSON serialisation, or None on failure.
        """
        try:
            import whisperx  # type: ignore
        except ImportError:
            logger.error(
                "whisperx is not installed. Run: pip install whisperx  (and install ffmpeg)"
            )
            return None

        try:
            device = self.config.whisper_device
            compute_type = "float16" if device == "cuda" else "float32"

            # --- 1. Transcribe ---
            if self._whisperx_model is None:
                logger.info(
                    "Loading WhisperX model '%s' on %s …",
                    self.config.whisper_model, device,
                )
                self._whisperx_model = whisperx.load_model(
                    self.config.whisper_model,
                    device,
                    compute_type=compute_type,
                    language=self.config.whisper_language or None,
                )

            audio = whisperx.load_audio(audio_path)
            result = self._whisperx_model.transcribe(audio, batch_size=16)
            detected_lang = result.get("language", "en")
            lang = self.config.whisper_language or detected_lang

            # --- 2. Align ---
            if lang not in self._align_models:
                model_a, metadata = whisperx.load_align_model(
                    language_code=lang, device=device
                )
                self._align_models[lang] = (model_a, metadata)

            model_a, metadata = self._align_models[lang]
            result = whisperx.align(
                result["segments"], model_a, metadata, audio, device,
                return_char_alignments=False,
            )

            # --- 3. Diarize (optional) ---
            hf_token = self.config.whisper_hf_token
            if hf_token:
                diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=hf_token, device=device
                )
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            else:
                logger.debug(
                    "No HF token → skipping speaker diarization for %s",
                    audio_path,
                )

            result["language"] = lang
            return result

        except Exception as exc:
            logger.warning("WhisperX error on %s: %s", audio_path, exc)
            return None
