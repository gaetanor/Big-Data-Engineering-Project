import asyncio
import re
import logging
from typing import Any, Dict, List, Set, Tuple
import yt_dlp
from .config import ScraperConfig

logger = logging.getLogger("instagram_scraper")

def generate_queries(topics: List[str], extra_keywords: List[str]) -> List[Tuple[str, str]]:
    queries = []
    for topic in topics:
        clean_topic = topic.strip("#").strip().lower().replace(" ", "")
        queries.append((clean_topic, topic))
        for kw in extra_keywords:
            clean_kw = kw.strip().lower().replace(" ", "")
            queries.append((f"{clean_topic}{clean_kw}", topic))
    return queries

class InstagramScraper:
    def __init__(self, config: ScraperConfig) -> None:
        self.config = config
        self._loop = asyncio.new_event_loop()
        self._context = None
        self._pw = None
        self._loop.run_until_complete(self._open_session())

    async def _open_session(self) -> None:
        import pathlib
        from playwright.async_api import async_playwright

        profile_dir = str(pathlib.Path(self.config.data_dir) / "browser_profile_ig")
        self._pw_cm = async_playwright()
        self._pw = await self._pw_cm.__aenter__()

        self._context = await self._pw.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False, # Lo teniamo visibile per il login manuale iniziale
            viewport={"width": 1280, "height": 800},
        )
        
        await self._ensure_logged_in()
        await self._export_cookies_file()

    async def _ensure_logged_in(self) -> None:
        page = await self._context.new_page()
        await page.goto("https://www.instagram.com/", wait_until="domcontentloaded")
        
        print("\n[IG] Verifica login. Se non sei loggato, accedi ora nel browser.")
        for _ in range(60): # 1 minuto per fare il login
            cookies = await self._context.cookies("https://www.instagram.com")
            if any(c["name"] == "sessionid" for c in cookies):
                print("[IG] Sessione attiva trovata!")
                break
            await asyncio.sleep(1)
        await page.close()

    async def _export_cookies_file(self) -> None:
        import pathlib
        cookies = await self._context.cookies("https://www.instagram.com")
        cookies_path = pathlib.Path(self.config.data_dir) / "cookies.txt"
        lines = ["# Netscape HTTP Cookie File\n"]
        for c in cookies:
            lines.append(f"{c['domain']}\tTRUE\t{c['path']}\tTRUE\t{int(c.get('expires', 0))}\t{c['name']}\t{c['value']}\n")
        cookies_path.write_text("".join(lines), encoding="utf-8")

    def fetch_reels_urls(self, hashtag: str, max_results: int) -> List[str]:
        return self._loop.run_until_complete(self._fetch_urls_async(hashtag, max_results))

    async def _fetch_urls_async(self, hashtag: str, max_results: int) -> List[str]:
        page = await self._context.new_page()
        urls = set()
        try:
            await page.goto(f"https://www.instagram.com/explore/tags/{hashtag}/")
            await page.wait_for_timeout(4000)
            
            for _ in range(10): # Scrolliamo per trovare più video
                await page.evaluate("window.scrollBy(0, 1000)")
                await page.wait_for_timeout(2000)
                
                hrefs = await page.evaluate('''() => {
                    return Array.from(document.querySelectorAll('a[href*="/reel/"], a[href*="/p/"]')).map(a => a.href);
                }''')
                
                urls.update(hrefs)
                if len(urls) >= max_results:
                    break
        finally:
            await page.close()
            
        return list(urls)[:max_results]

    def fetch_full_info(self, url: str) -> Dict[str, Any]:
        """Usa yt-dlp con i cookie per estrarre tutti i metadati del Reel."""
        import pathlib
        cookies_path = pathlib.Path(self.config.data_dir) / "cookies.txt"
        opts = {
            "quiet": True,
            "no_warnings": True,
            "cookiefile": str(cookies_path),
            "extract_flat": False,
            "getcomments": True
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except Exception:
            return {}

    def close(self):
        self._loop.run_until_complete(self._context.close())

    def get_extra_data(self, url: str, thumbnail_url: str = "") -> Dict[str, Any]:
        """Wrapper sincrono per recuperare commenti e visualizzazioni."""
        return self._loop.run_until_complete(self._fetch_extra_data_async(url, thumbnail_url))

    async def _fetch_extra_data_async(self, url: str, thumbnail_url: str = "") -> Dict[str, Any]:
        all_comments: List[Dict[str, Any]] = []
        view_count = None
        page = await self._context.new_page()
        
        print(f"  [DEBUG] Navigazione verso: {url}")

        # 1. INTERCETTATORE DI RETE: Algoritmo Ricorsivo Universale
        async def on_response(response):
            nonlocal view_count
            
            if "comment/list" in response.url or "api/comment" in response.url:
                try:
                    data = await response.json()
                    for c in data.get("comments", []):
                        all_comments.append({
                            "comment_id": str(c.get("cid", "")),
                            "text": c.get("text", ""),
                            "author_id": c.get("user", {}).get("unique_id") or "anon",
                            "like_count": c.get("digg_count") or 0
                        })
                except Exception:
                    pass

            if response.ok and response.request.resource_type in ["fetch", "xhr"]:
                try:
                    data = await response.json()
                    
                    def extract_views_recursive(obj):
                        if isinstance(obj, dict):
                            for k, v in obj.items():
                                if k in ["play_count", "video_view_count"] and isinstance(v, int):
                                    return v
                                res = extract_views_recursive(v)
                                if res is not None:
                                    return res
                        elif isinstance(obj, list):
                            for item in obj:
                                res = extract_views_recursive(item)
                                if res is not None:
                                    return res
                        return None

                    found_views = extract_views_recursive(data)
                    if found_views and view_count is None:
                        view_count = found_views
                        print(f"  [DEBUG-RETE] BOOM! Trovato view_count ricorsivo: {view_count}")
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            # 2. API ENDPOINT INTERNO
            try:
                api_url = url.rstrip('/') + "/?__a=1&__d=dis"
                api_response = await self._context.request.get(api_url)
                if api_response.ok:
                    api_json = await api_response.json()
                    items = api_json.get("items", [])
                    if items and not view_count:
                        view_count = items[0].get("play_count") or items[0].get("video_view_count")
                    if not view_count:
                        graphql = api_json.get("graphql", {}).get("shortcode_media", {})
                        view_count = graphql.get("video_view_count") or graphql.get("play_count")
                    if view_count:
                        print(f"  [DEBUG-API] view_count estratto dall'API nativa: {view_count}")
            except Exception:
                pass

            # 3. CARICAMENTO REEL
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            except Exception:
                pass
                
            await page.wait_for_timeout(4000)
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1000)

            # 4. SCROLL E DOM SCRAPING (Commenti)
            try:
                await page.mouse.move(800, 400)
                for _ in range(10):
                    await page.mouse.wheel(0, 1500)
                    await page.wait_for_timeout(1000)
            except Exception:
                pass

            print("  [DEBUG-DOM] Estrazione commenti in corso...")
            comments_data = await page.evaluate('''() => {
                let results = [];
                let elements = Array.from(document.querySelectorAll('span, div, button'));
                let replyNodes = elements.filter(el => {
                    let t = el.innerText ? el.innerText.trim() : "";
                    return t === 'Reply' || t === 'Rispondi';
                });

                replyNodes.forEach(node => {
                    let container = node.closest('li');
                    if (!container) {
                        container = node.parentElement;
                        for(let i=0; i<3; i++) {
                            if(container) container = container.parentElement;
                        }
                    }

                    if (container) {
                        let authorEl = container.querySelector('h2, h3, a');
                        let author = authorEl ? authorEl.innerText.trim() : "";

                        let spans = Array.from(container.querySelectorAll('span[dir="auto"], a'));
                        let texts = spans.map(s => s.innerText.trim()).filter(t => t.length > 0);
                        
                        let commentText = "";
                        let likeCount = 0;

                        texts.forEach(t => {
                            // 1. Scarta il nome utente (che spesso IG renderizza due volte di fila)
                            if (t === author || t === author + " " + author) return;

                            // 2. Intercetta i Mi Piace (es. "1 Mi piace", "Mi piace: 2", "15K likes")
                            let likeMatch = t.match(/(?:Mi piace|Likes?)\\s*:\\s*([\\d\\.,]+[kmKM]?)|([\\d\\.,]+[kmKM]?)\\s*(?:Mi piace|Likes?)/i);
                            if (likeMatch) {
                                let rawLikes = (likeMatch[1] || likeMatch[2]).toLowerCase().replace(/[,.]/g, '');
                                if (rawLikes.includes('k')) likeCount = parseFloat(rawLikes) * 1000;
                                else if (rawLikes.includes('m')) likeCount = parseFloat(rawLikes) * 1000000;
                                else likeCount = parseInt(rawLikes) || 0;
                                return; // Esce dal ciclo per non incollare la scritta dei like nel testo!
                            }

                            // 3. Filtra via i pulsanti UI e le date
                            let isUI = ["Reply", "Rispondi", "Like", "Mi piace", "Vedi traduzione", "See translation", "Audio originale", "Original audio"].includes(t);
                            let isDate = t.match(/^\\d+\\s*(sett|g|h|m|s|w|d|ore|giorni)$/i);
                            
                            // Se è testo reale, lo aggiunge al commento
                            if (!isUI && !isDate) {
                                commentText += t + " ";
                            }
                        });

                        commentText = commentText.trim();
                        let timeEl = container.querySelector('time');
                        let createdAt = timeEl ? timeEl.getAttribute('datetime') : "";

                        if (commentText.length > 1 && author && !author.includes("Meta") && !author.includes("Home")) {
                            results.push({
                                comment_id: Math.random().toString(36).substr(2, 9), // ID fittizio per i commenti DOM
                                text: commentText,
                                author_id: author,
                                like_count: likeCount, // Ora popolerà i like corretti
                                created_at: createdAt || ""
                            });
                        }
                    }
                });
                return results;
            }''')

            for c in comments_data:
                all_comments.append(c)

            # ----------------------------------------------------------------------
            # IL SUGGERIMENTO DEL TUO VIDEO: PIANO C (Ricerca nella griglia del Profilo)
            # ----------------------------------------------------------------------
            if view_count is None:
                import re
                # Estraiamo l'ID esatto del Reel dall'URL (es. 'DJlV5PnzoBI')
                vid_match = re.search(r'/(?:p|reel)/([^/?#]+)', url)
                video_id = vid_match.group(1) if vid_match else None

                if video_id:
                    # Troviamo lo username del CREATORE del reel.
                    # APPROCCIO DEFINITIVO: cerchiamo l'<a> con href="/username/" che si trova
                    # nel contenitore del post (colonna destra con header+commenti).
                    # Instagram struttura sempre il post come:
                    #   article > [sezione-video | sezione-info]
                    #             sezione-info > header > ... > a[href="/username/"]
                    # Partiamo dall'elemento più specifico e risaliamo, oppure usiamo
                    # l'URL della pagina stessa come ancora: l'URL contiene /p/<id> o /reel/<id>
                    # e il link autore è SEMPRE il primo /username/ che appare dentro
                    # l'elemento <article> o nel main content, NON nella sidebar sinistra.
                    username = await page.evaluate('''() => {
                        const reserved = new Set([
                            'explore', 'reels', 'direct', 'stories', 'accounts',
                            'p', 'tv', 'reel', 'live', 'locations', 'tags', 'about',
                            'privacy', 'help', 'press', 'api', 'blog', 'jobs',
                            'legal', 'directory', 'lite', 'features', 'meta'
                        ]);

                        function extractUsername(href) {
                            if (!href) return null;
                            const m = href.match(/^\/([A-Za-z0-9_.]+)\/?$/);
                            if (m && !reserved.has(m[1].toLowerCase())) return m[1];
                            return null;
                        }

                        // STRATEGIA 1 (più affidabile): cerca dentro <article>
                        // Instagram wrappa il post in un <article>, che NON include la sidebar.
                        const article = document.querySelector('article');
                        if (article) {
                            const links = Array.from(article.querySelectorAll('a[href]'));
                            for (const a of links) {
                                const u = extractUsername(a.getAttribute('href'));
                                if (u) return u;
                            }
                        }

                        // STRATEGIA 2: cerca dentro <main> (esclude sidebar sinistra di IG
                        // che è fuori da <main> nella struttura IG desktop)
                        const main = document.querySelector('main');
                        if (main) {
                            const links = Array.from(main.querySelectorAll('a[href]'));
                            for (const a of links) {
                                const u = extractUsername(a.getAttribute('href'));
                                if (u) return u;
                            }
                        }

                        // STRATEGIA 3 (ultimo fallback): qualsiasi link /username/ nell'intera pagina
                        // escludendo esplicitamente l'account loggato leggendolo dall'URL del profilo
                        // (che IG inietta nella sidebar come link alla propria pagina).
                        const selfLinks = Array.from(document.querySelectorAll(
                            'nav a[href], [role="navigation"] a[href], [role="banner"] a[href]'
                        ));
                        const selfHrefs = new Set(selfLinks.map(a => a.getAttribute('href')));

                        const allLinks = Array.from(document.querySelectorAll('a[href]'));
                        for (const a of allLinks) {
                            const href = a.getAttribute('href') || '';
                            if (selfHrefs.has(href)) continue;  // salta link della sidebar/nav
                            const u = extractUsername(href);
                            if (u) return u;
                        }

                        return null;
                    }''')

                    if username:
                        print(f"  [DEBUG-PROFILE] Tentativo Estremo: navigazione su /{username}/reels/ ...")
                        try:
                            # Navighiamo verso la tab Reels del creatore
                            profile_url = f"https://www.instagram.com/{username}/reels/"
                            await page.goto(profile_url, wait_until="domcontentloaded", timeout=15000)
                            # Aspettiamo che almeno una copertina reel sia nel DOM prima di scrollare
                            try:
                                await page.wait_for_selector('a[href*="/reel/"]', timeout=6000)
                            except Exception:
                                pass
                            await page.wait_for_timeout(2000)

                            found_views = None
                            # Facciamo fino a 50 scroll per cercare il video nella griglia
                            for scroll_i in range(50):
                                # Cerchiamo il link corrispondente al nostro reel con DUE strategie:
                                # 1. href contenente lo shortcode dell'URL (es. "DWPvuIFCNh7")
                                # 2. href contenente la thumbnail_url dal JSON (ancora alternativa)
                                # Dopo ogni scroll aspettiamo che nuovi nodi siano nel DOM.
                                grid_text = await page.evaluate('''([vid, thumbUrl]) => {
                                    // STRATEGIA 1: shortcode nell'href del link copertina
                                    let aTag = document.querySelector(`a[href*="${vid}"]`);

                                    // STRATEGIA 2: thumbnail URL come ancora
                                    // Le copertine nella griglia hanno un <img src="..."> che contiene
                                    // lo stesso ID numerico presente nella thumbnail_url di yt-dlp.
                                    if (!aTag && thumbUrl) {
                                        // Estraiamo l'ID numerico dalla thumbnail (es. "123456789_n.jpg" -> "123456789")
                                        const thumbMatch = thumbUrl.match(/\/(\d{10,})_/);
                                        if (thumbMatch) {
                                            const numId = thumbMatch[1];
                                            // Cerchiamo un <img> con src contenente quell'ID numerico
                                            const imgs = Array.from(document.querySelectorAll('article img, main img'));
                                            for (const img of imgs) {
                                                if ((img.src || '').includes(numId)) {
                                                    // Risaliamo all'<a> genitore che wrappa la copertina
                                                    let cur = img.parentElement;
                                                    for (let i = 0; i < 6; i++) {
                                                        if (!cur) break;
                                                        if (cur.tagName === 'A' && cur.href) { aTag = cur; break; }
                                                        cur = cur.parentElement;
                                                    }
                                                    break;
                                                }
                                            }
                                        }
                                    }

                                    if (aTag) return aTag.innerText.trim();
                                    return null;
                                }''', [video_id, thumbnail_url])

                                if grid_text:
                                    # Spesso il testo contiene "Fissato in alto \n 152.000". Prendiamo l'ultima riga.
                                    lines = grid_text.split('\n')
                                    target_text = lines[-1].strip()
                                    
                                    # Regex aggiornata: gestisce spazi e suffissi lunghi come "Mln", "mila", "K"
                                    v_match = re.search(r'([\d\.,]+)\s*([kKmM][a-zA-Z]*)?', target_text)
                                    if v_match:
                                        num_str = v_match.group(1)
                                        suffix = (v_match.group(2) or "").lower()

                                        try:
                                            if 'm' in suffix:  # 'm', 'mln', 'milioni'
                                                num_str = num_str.replace(',', '.')
                                                found_views = int(float(num_str) * 1_000_000)
                                            elif 'k' in suffix or 'mila' in suffix:  # 'k', 'mila'
                                                num_str = num_str.replace(',', '.')
                                                found_views = int(float(num_str) * 1_000)
                                            else:
                                                # Nessun suffisso: gestisce separatori migliaia italiani
                                                # Es: "1.500" -> 1500, "1.152.000" -> 1152000, "1.234,5" -> 1234
                                                if ',' in num_str and '.' in num_str:
                                                    # "1.234,5" -> rimuovi punti, virgola = decimale
                                                    clean_num = num_str.replace('.', '').replace(',', '.')
                                                elif ',' in num_str:
                                                    parts = num_str.split(',')
                                                    if len(parts) == 2 and len(parts[1]) <= 2:
                                                        clean_num = num_str.replace(',', '.')  # decimale
                                                    else:
                                                        clean_num = num_str.replace(',', '')   # migliaia
                                                else:
                                                    # Solo punti: sep migliaia se tutte le parti dopo il primo hanno 3 cifre
                                                    parts = num_str.split('.')
                                                    if len(parts) > 1 and all(len(p) == 3 for p in parts[1:]):
                                                        clean_num = num_str.replace('.', '')
                                                    else:
                                                        clean_num = num_str
                                                found_views = int(float(clean_num))
                                        except ValueError:
                                            pass
                                    break # Interrompe il ciclo for, l'abbiamo trovato!
                                # Se non l'abbiamo ancora visto, scrolliamo giù
                                await page.mouse.wheel(0, 2000)
                                await page.wait_for_timeout(1000)

                            if found_views is not None:
                                view_count = found_views
                                print(f"  [DEBUG-PROFILE] view_count estratto dalla copertina: {view_count}")
                            else:
                                print("  [DEBUG-PROFILE] Reel troppo vecchio o non individuato nella griglia.")

                        except Exception as e:
                            print(f"  [DEBUG-PROFILE] Errore durante la navigazione profilo: {e}")

        except Exception as exc:
            print(f"  [DEBUG-ERRORE] Fallimento critico estrazione IG: {exc}")
        finally:
            await page.close()

        # Deduplicazione commenti
        seen = set()
        clean_comments = []
        for c in all_comments:
            if c["text"] not in seen:
                seen.add(c["text"])
                clean_comments.append(c)
                
        if len(clean_comments) > 0 and ("#" in clean_comments[0]["text"] or len(clean_comments[0]["text"]) > 200):
            clean_comments.pop(0)

        print(f"  [DEBUG] Estrazione IG finita. Views catturate: {view_count}, Commenti: {len(clean_comments)}")
        return {"comments": clean_comments[:50], "view_count": view_count}
    
    def parse_record(self, info: Dict[str, Any], topic: str, extra_data: Dict[str, Any]) -> Dict[str, Any]:
        desc = info.get("description", "")
        
        final_view_count = extra_data.get("view_count")
        if final_view_count is None:
            final_view_count = info.get("view_count")
            
        final_comments = []
        raw_yt_comments = info.get("comments") or []
        
        if len(raw_yt_comments) > 0:
            for c in raw_yt_comments[:50]:
                final_comments.append({
                    "comment_id": str(c.get("id", "")),
                    "text": c.get("text", ""),
                    "author_id": c.get("author", "") or c.get("author_id", "anon"),
                    "like_count": c.get("like_count", 0),
                    "created_at": str(c.get("timestamp", ""))
                })
        else:
            final_comments = extra_data.get("comments", [])
            
        # FIX COMMENT COUNT: yt-dlp restituisce '0' invece di 'None' quando fallisce
        final_comment_count = info.get("comment_count")
        if final_comment_count is None or (final_comment_count == 0 and len(final_comments) > 0):
            final_comment_count = len(final_comments)

        # --- FIX INSTAGRAM: Caption as Title ---
        raw_title = info.get("title") or ""
        uploader = info.get("uploader", "")
        
        if "instagram" in info.get("extractor", "").lower() or raw_title.startswith("Video by"):
            if desc:
                # Splittiamo per a capo e prendiamo la prima riga non vuota
                lines = [line.strip() for line in desc.split('\n') if line.strip()]
                if lines:
                    first_line = lines[0]
                    # Tagliamo a 80 caratteri
                    final_title = first_line[:80] + ("..." if len(first_line) > 80 else "")
                else:
                    final_title = f"Instagram Reel da {uploader}"
            else:
                final_title = f"Instagram Reel da {uploader}"
        else:
            final_title = raw_title
        # ---------------------------------------

        return {
            "video_id": info.get("id", ""),
            "url": info.get("webpage_url", ""),
            "title": final_title,
            "description": desc,
            "duration": info.get("duration"),
            "view_count": final_view_count,
            "like_count": info.get("like_count"),
            "comment_count": final_comment_count,
            "uploader": uploader,
            "channel_id": info.get("uploader_id", ""),
            "channel_url": info.get("uploader_url", ""),
            "upload_date": info.get("upload_date", ""),
            "hashtags": re.findall(r"#\w+", desc),
            "thumbnail_url": info.get("thumbnail", ""),
            "comments": final_comments,
            "topic": topic,
            "platform": "instagram_reels",
            "collected_at": info.get("upload_date", "")
        }