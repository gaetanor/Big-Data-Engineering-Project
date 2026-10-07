from __future__ import annotations

import asyncio
import json as _json
import logging
import re
import time as _time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import ScraperConfig

logger = logging.getLogger("tiktok_scraper")


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

_HASHTAG_TEMPLATES = [
    "{topic}",
    "{topic}shorts",
    "{topic}viral",
    "{topic}fyp",
    "{topic}video",
    "{topic}trending",
]


def generate_queries(
    topics: List[str], extra_keywords: List[str]
) -> List[Tuple[str, str]]:
    queries: List[Tuple[str, str]] = []
    for raw_topic in topics:
        topic = raw_topic.strip("#").strip().lower().replace(" ", "")
        for tmpl in _HASHTAG_TEMPLATES:
            queries.append((tmpl.format(topic=topic), raw_topic))
        for kw in extra_keywords:
            kw_clean = kw.strip().lower().replace(" ", "")
            queries.append((f"{topic}{kw_clean}", raw_topic))
            queries.append((kw_clean, raw_topic))
    return queries


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

class TikTokScraper:
    """
    Pure-Playwright scraper: opens a real webkit browser, navigates to each
    TikTok hashtag page, and intercepts the JSON responses that TikTok's own
    frontend fetches to populate the feed.

    No developer account, no API key, no TikTokApi library required.
    If you have a TikTok account you can optionally pass --session-id for
    higher rate limits, but it is not mandatory.

    Requirements (one-time setup):
        pip install playwright
        python -m playwright install webkit
    """

    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self._loop = asyncio.new_event_loop()
        self._pw_cm = None
        self._pw = None
        self._context = None  # BrowserContext (launch_persistent_context returns context directly)
        self._bg_page = None
        self._loop.run_until_complete(self._open_session())

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def _open_session(self) -> None:
        import pathlib

        try:
            from playwright.async_api import async_playwright  # type: ignore
        except ImportError:
            raise ImportError(
                "playwright non installato.\n"
                "Eseguire:\n"
                "  pip install playwright\n"
                "  python -m playwright install chromium"
            )

        # Persistent profile directory — TikTok sees a real browser with history,
        # fonts and localStorage intact, so fingerprinting checks pass.
        profile_dir = str(pathlib.Path(self.config.data_dir) / "browser_profile")

        self._pw_cm = async_playwright()
        self._pw = await self._pw_cm.__aenter__()

        # launch_persistent_context returns a BrowserContext directly (no separate Browser object)
        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
            timezone_id="America/New_York",
        )

        # Optionally inject msToken cookie (anonymous visitor token)
        if self.config.ms_token:
            await self._context.add_cookies([{
                "name": "msToken", "value": self.config.ms_token,
                "domain": ".tiktok.com", "path": "/",
            }])

        # Check if already logged in; if not, prompt for manual login
        await self._ensure_logged_in()

        # Export cookies to a flat file so yt-dlp can read them
        # without trying to access the locked Chromium SQLite database.
        await self._export_cookies_file()

        # Persistent blank page: keeps the browser window visible between fetches
        self._bg_page = await self._context.new_page()
        await self._bg_page.goto("about:blank")

    async def _export_cookies_file(self) -> None:
        """
        Write TikTok cookies to data/cookies.txt in Netscape format.
        yt-dlp reads this file directly — no access to the locked SQLite DB needed.
        """
        import pathlib
        cookies = await self._context.cookies("https://www.tiktok.com")
        cookies_path = pathlib.Path(self.config.data_dir) / "cookies.txt"
        cookies_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Netscape HTTP Cookie File\n"]
        for c in cookies:
            domain = c["domain"]
            include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
            path = c.get("path", "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = int(c["expires"]) if c.get("expires", -1) > 0 else 0
            lines.append(
                f"{domain}\t{include_subdomains}\t{path}\t{secure}\t{expires}\t{c['name']}\t{c['value']}\n"
            )
        cookies_path.write_text("".join(lines), encoding="utf-8")
        logger.info("Cookie TikTok esportati → %s", cookies_path)

    async def _is_logged_in(self) -> bool:
        """Return True if a valid sessionid cookie is already present."""
        cookies = await self._context.cookies("https://www.tiktok.com")
        sid = next((c["value"] for c in cookies if c["name"] == "sessionid"), "")
        return bool(sid)

    async def _ensure_logged_in(self) -> None:
        """
        If the persistent profile has no sessionid, open TikTok's *home page*
        (not the /login URL — TikTok is less suspicious of organic navigation)
        and wait up to 3 minutes for the user to log in manually.

        The session is automatically preserved in the persistent profile for
        all future runs — no manual cookie extraction needed.
        """
        if await self._is_logged_in():
            logger.info("Sessione TikTok già attiva nel profilo persistente.")
            return

        print("\n" + "=" * 62, flush=True)
        print("[LOGIN] Nessuna sessione TikTok trovata nel profilo salvato.", flush=True)
        print("[LOGIN] Si aprirà TikTok nel browser. Accedi manualmente.", flush=True)
        print("[LOGIN] Lo script riprende automaticamente dopo il login.", flush=True)
        print("[LOGIN] La sessione verrà salvata per le esecuzioni future.", flush=True)
        print("=" * 62 + "\n", flush=True)

        login_page = await self._context.new_page()
        # Navigate to the home page, not /login — less bot-like, user clicks login themselves
        await login_page.goto("https://www.tiktok.com", wait_until="domcontentloaded")

        for _ in range(180):          # wait up to 3 minutes
            await login_page.wait_for_timeout(1_000)
            if await self._is_logged_in():
                print("[LOGIN] Login completato! Sessione salvata nel profilo persistente.", flush=True)
                try:
                    await login_page.close()
                except Exception:
                    pass
                return

        print("[LOGIN] Timeout (3 min) — si procede senza login.", flush=True)
        try:
            await login_page.close()
        except Exception:
            pass

    def close(self) -> None:
        async def _close():
            if self._bg_page and not self._bg_page.is_closed():
                await self._bg_page.close()
            if self._context:
                await self._context.close()
            if self._pw_cm:
                await self._pw_cm.__aexit__(None, None, None)

        if not self._loop.is_closed():
            try:
                self._loop.run_until_complete(_close())
            except Exception as exc:
                logger.warning("Errore chiusura browser: %s", exc)
            self._loop.close()

    # ------------------------------------------------------------------
    # Public interface (synchronous facade)
    # ------------------------------------------------------------------

    def fetch_with_backoff(self, hashtag: str) -> List[Dict[str, Any]]:
        """Fetch videos from a TikTok hashtag page with exponential backoff."""
        n = self.config.results_per_query
        wait = self.config.backoff_base

        for attempt in range(1, self.config.max_retries + 1):
            try:
                return self._loop.run_until_complete(
                    self._fetch_hashtag(hashtag, n)
                )
            except Exception as exc:
                msg = str(exc).lower()
                is_rate_limit = "rate" in msg or "429" in msg or "too many" in msg
                if is_rate_limit:
                    logger.warning(
                        "Rate limited attempt %d/%d. Sleeping %.0fs …",
                        attempt, self.config.max_retries, wait,
                    )
                    _time.sleep(wait)
                    wait = min(wait * 2, self.config.backoff_max_wait)
                else:
                    logger.error("Errore tentativo %d per #%s: %s", attempt, hashtag, exc)
                    return []

        logger.error("Tutti i retry esauriti per hashtag: #%s", hashtag)
        return []

    def is_duration_candidate(self, entry: Dict[str, Any]) -> bool:
        """Phase 1: quick duration filter on flat metadata."""
        duration = (
            entry.get("video", {}).get("duration")
            or entry.get("duration")
        )
        if duration is None:
            return True
        try:
            return int(duration) <= self.config.max_duration
        except (TypeError, ValueError):
            return True

    def parse_record(
        self, entry: Dict[str, Any], search_query: str, topic: str = ""
    ) -> Optional[Dict[str, Any]]:
        """Map a raw TikTok API item dict to a normalized record dict."""
        video_id = str(entry.get("id") or "").strip()
        if not video_id:
            return None

        video_info = entry.get("video", {})
        duration = video_info.get("duration") or entry.get("duration")
        try:
            duration = int(duration) if duration is not None else None
        except (TypeError, ValueError):
            duration = None
        if duration is not None and duration > self.config.max_duration:
            return None

        author = entry.get("author", {})
        username = author.get("uniqueId", "")
        url = f"https://www.tiktok.com/@{username}/video/{video_id}"

        description = str(entry.get("desc") or "")
        if len(description) > 5000:
            description = description[:5000]

        stats = entry.get("stats", {})
        music = entry.get("music", {})

        create_time = entry.get("createTime")
        upload_date = ""
        if create_time:
            try:
                upload_date = datetime.utcfromtimestamp(int(create_time)).strftime("%Y%m%d")
            except (TypeError, ValueError):
                pass

        return {
            "video_id": video_id,
            "url": url,
            "shorts_url": url,
            "title": description,
            "description": description,
            "duration": duration,
            "view_count": stats.get("playCount") or stats.get("viewCount"),
            "like_count": stats.get("diggCount") or stats.get("likeCount"),
            "comment_count": stats.get("commentCount"),
            "share_count": stats.get("shareCount"),
            "uploader": author.get("nickname") or "",
            "channel_id": username,
            "channel_url": f"https://www.tiktok.com/@{username}" if username else "",
            "upload_date": upload_date,
            "language": entry.get("language") or "",
            "tags": [],
            "categories": [],
            "hashtags": self._extract_hashtags(description),
            "music_track": music.get("title") or "",
            "music_author": music.get("authorName") or "",
            "music_id": str(music.get("id") or ""),
            "thumbnail_url": video_info.get("cover") or "",
            "comments": [],
            "topic": topic,
            "platform": "tiktok",
            "search_query": search_query,
            "collected_at": datetime.now(timezone.utc).isoformat(),
        }

    def get_comments(self, video_id: str, url: str) -> List[Dict[str, Any]]:
        """Synchronous wrapper around _fetch_comments_async."""
        return self._loop.run_until_complete(
            self._fetch_comments_async(video_id, url)
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _fetch_comments_async(self, video_id: str, url: str) -> List[Dict[str, Any]]:
        all_comments: List[Dict[str, Any]] = []
        page = await self._context.new_page()
        
        print(f"  [DEBUG] Navigazione verso: {url}")

        # PIANO A: Monitoraggio Rete
        async def on_response(response):
            if "comment/list" in response.url or "api/comment" in response.url:
                try:
                    data = await response.json()
                    batch = data.get("comments") or []
                    if batch:
                        print(f"  [DEBUG-RETE] Intercettata API! Trovati: {len(batch)}")
                    for c in batch:
                        all_comments.append({
                            "comment_id": str(c.get("cid", "")),
                            "text": c.get("text", ""),
                            "author_id": c.get("user", {}).get("unique_id") or "anon",
                            "like_count": c.get("digg_count") or 0
                        })
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            await page.goto(url, wait_until="networkidle", timeout=20000)
            print("  [DEBUG] Rete stabilizzata. Preparazione interfaccia...")
            await page.wait_for_timeout(2000)

            # 1. RIMUOVERE OSTACOLI (Chiude il banner degli shortcut per liberare lo schermo)
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1000)

            # 2. APRIRE I COMMENTI
            try:
                await page.click('[data-e2e="comment-icon"]', timeout=3000)
                print("  [DEBUG] Cliccato sull'icona dei commenti.")
            except Exception:
                try:
                    await page.click('text="Comments"', timeout=3000)
                    print("  [DEBUG] Cliccato sul tab 'Comments'.")
                except Exception:
                    print("  [DEBUG] Pulsante commenti non trovato.")

            await page.wait_for_timeout(3000)

            # 3. SCROLL SUI COMMENTI
            try:
                # Troviamo il primo commento e posizioniamo il mouse FISICAMENTE sopra di esso
                first_comment = page.locator('[data-e2e="comment-level-1"]').first
                await first_comment.wait_for(timeout=4000)
                await first_comment.hover()
                print("  [DEBUG] Mouse ancorato alla sezione commenti. Avvio scroll della rotellina...")
                
                # Facciamo scorrere la rotellina del mouse (simula perfettamente l'utente e non cambia video)
                for _ in range(4):
                    await page.mouse.wheel(0, 2000) # Scorre verso il basso di 2000 pixel
                    await page.wait_for_timeout(1500)
            except Exception as e:
                print(f"  [DEBUG] Nessun commento a cui ancorare il mouse (forse video senza commenti).")

        except Exception as exc:
            print(f"  [DEBUG-ERRORE] Fallimento critico estrazione: {exc}")
        finally:
            await page.close()

        # Pulizia e Deduplicazione
        seen = set()
        unique_comments = []
        for c in all_comments:
            if c["text"] and c["text"] not in seen:
                seen.add(c["text"])
                unique_comments.append(c)

        print(f"  [DEBUG] Estrazione finita. Commenti unici finali: {len(unique_comments)}")
        return unique_comments[:50]

    @staticmethod
    def _extract_hashtags(text: str) -> List[str]:
        return re.findall(r"#\w+", text)

    async def _fetch_hashtag(self, hashtag: str, count: int) -> List[Dict[str, Any]]:
        """
        Two-source collection strategy:

        Source A — embedded JSON: TikTok pre-loads the first batch of videos
          inside a <script id="__UNIVERSAL_DATA_FOR_REHYDRATION__"> tag.
          Parsed immediately after DOM loads, no network interception needed.

        Source B — network interception: any JSON response from TikTok that
          contains an itemList / items key (no URL pattern filter, so it works
          regardless of TikTok's current API routing).

        Both sources are deduplicated by video id.
        """
        collected: List[Dict[str, Any]] = []
        seen_ids: Set[str] = set()

        def add_items(items: List[Dict[str, Any]]) -> None:
            for item in items:
                vid = str(item.get("id") or "").strip()
                if vid and vid not in seen_ids:
                    seen_ids.add(vid)
                    collected.append(item)

        page = await self._context.new_page()
        await page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});
            window.chrome = { runtime: {} };
        """)

        # Source B: intercept ALL JSON responses, look for itemList / items
        async def on_response(response) -> None:
            if self.config.verbose:
                print(f"[NET] {response.status} {response.url[:100]}", flush=True)
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return
            try:
                data = await response.json()
            except Exception:
                return
            items = data.get("itemList") or data.get("items") or []
            if items:
                print(
                    f"[DEBUG] {response.url[:80]}  →  {len(items)} item",
                    flush=True,
                )
                add_items(items)

        page.on("response", on_response)

        try:
            await page.goto(
                f"https://www.tiktok.com/tag/{hashtag}",
                wait_until="domcontentloaded",
                timeout=30_000,
            )
            await page.wait_for_timeout(3_000)

            # Detect login redirect
            final_url = page.url
            if "/login" in final_url:
                await page.close()
                logger.warning(
                    "Reindirizzato al login per #%s. "
                    "Il profilo persistente potrebbe essere scaduto. "
                    "Riavvia lo script per effettuare nuovamente il login.",
                    hashtag,
                )
                return []

            if self.config.verbose:
                title = await page.title()
                print(f"[PAGE] url={final_url}  title={title!r}", flush=True)

            # Salva screenshot diagnostico per il primo hashtag
            if self.config.verbose:
                import pathlib
                ss_path = pathlib.Path(self.config.data_dir) / f"debug_{hashtag}.png"
                ss_path.parent.mkdir(parents=True, exist_ok=True)
                await page.screenshot(path=str(ss_path), full_page=False)
                print(f"[DEBUG] Screenshot salvato → {ss_path}", flush=True)

            # Source A: embedded JSON in the page HTML
            try:
                raw = await page.evaluate(
                    "() => {"
                    "  const el = document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__');"
                    "  return el ? el.textContent : null;"
                    "}"
                )
                if raw:
                    scope = _json.loads(raw).get("__DEFAULT_SCOPE__", {})
                    for value in scope.values():
                        if not isinstance(value, dict):
                            continue
                        candidates = [
                            value.get("itemList"),
                            value.get("videoList"),
                            (value.get("challengeInfo") or {}).get("itemList"),
                        ]
                        for items in candidates:
                            if items:
                                add_items(items)
                                print(
                                    f"[DEBUG] Embedded JSON: {len(items)} item",
                                    flush=True,
                                )
                else:
                    print("[DEBUG] __UNIVERSAL_DATA_FOR_REHYDRATION__ non trovato nel DOM", flush=True)
            except Exception as exc:
                print(f"[DEBUG] Embedded JSON extraction failed: {exc}", flush=True)

            print(
                f"[#]{hashtag}: {len(collected)} video trovati dopo il caricamento iniziale",
                flush=True,
            )

            # Scroll to trigger lazy loading of additional videos
            prev_count = 0
            stall_rounds = 0
            while len(collected) < count and stall_rounds < 5:
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2_000)
                if len(collected) == prev_count:
                    stall_rounds += 1
                else:
                    stall_rounds = 0
                    print(
                        f"[#]{hashtag}: {len(collected)} video (scroll in corso…)",
                        flush=True,
                    )
                prev_count = len(collected)

        finally:
            await page.close()

        return collected[:count]
