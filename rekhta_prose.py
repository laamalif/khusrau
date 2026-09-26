#!/usr/bin/env python3
"""Rekhta prose (afsana) fetcher. Runs in the Playwright container on `vagrant`.

Prose lives in a different URL space from verse: `/authors/<slug>` not `/poets/<slug>`,
and items are `/<kind>/<title-slug>-<author-slug>-<kind>`, e.g.
`/stories/toba-tek-singh-saadat-hasan-manto-stories`.

Extraction, established by probing (2026-09-04):
  - The story body is `div.pMC.showTranslation`. Selecting plain `div.pMC` gets the
    *promoted sher widget* instead -- it is the first pMC in the document and carries
    roman text, so a naive selector silently returns one couplet by another poet.
  - Inside it, each `div.w` is one PARAGRAPH (42 of them for Toba Tek Singh). On verse
    pages a `div.w` is a couplet holding `p` per hemistich, so both shapes are handled:
    `p` children if present, else the div's own text.
  - `div.overme` on a story page is an editorial blurb about the Urdu afsana in general,
    not the author's text. It sits outside `pMC.showTranslation`, so scoping to that
    container excludes it -- do not widen the selector.

Rights note, because this corpus is mostly not public domain. Roughly, life+60 India /
life+50 Pakistan: **Manto (d. 1955) and Premchand (d. 1936) are clear.** Krishan Chandar
(1977), Qudratullah Shahab (1986), Ismat Chughtai (1991), Mumtaz Mufti (1995), Ashfaq
Ahmed (2004), Bano Qudsia (2017) are **not** -- fine to read for alignment experiments,
not publishable as a released gold set. See FINDINGS §11.1.

Usage (inside container):
  rekhta_prose.py kinds  <author-slug>
  rekhta_prose.py index  <author-slug> [kind|all]
  rekhta_prose.py fetch  <author-slug> <outdir> [kind|all]   [REKHTA_WORKERS=4]
"""
import asyncio
import html.parser
import os
import re
import sys
import urllib.request

BASE = "https://www.rekhta.org"
WORKERS = int(os.environ.get("REKHTA_WORKERS", "4"))
NAV_TIMEOUT = int(os.environ.get("REKHTA_TIMEOUT", "60000"))
SEL_TIMEOUT = int(os.environ.get("REKHTA_SEL_TIMEOUT", "15000"))
PAUSE = float(os.environ.get("REKHTA_PAUSE", "0.4"))
URDU = re.compile(r"[؀-ۿ]")

# Discovered on /authors/saadat-hasan-manto. `quotes`, `video`, `gallery`, `blogs`,
# `ebooks`, `profile` are excluded: not primary prose text by this author.
KINDS = ("stories", "afsanche", "novelette", "articles", "khaka", "drama",
         "tanz-o-mazah", "tarajim", "translation", "other")

# Rekhta splits people into two URL spaces: prose writers under /authors/, poets under
# /poets/. Ghalib is a poet, so his LETTERS -- the reference that unblocks the 1888
# `kulliyatinasrghalib` lithograph (FINDINGS §21.5) -- are at /poets/mirza-ghalib/letters
# and need this prose extractor, not the verse one. Hence the space switch.
SPACE = os.environ.get("REKHTA_SPACE", "authors")   # authors | poets

BLOCK = re.compile(r"\.(png|jpe?g|gif|webp|svg|woff2?|ttf|eot|mp4|mp3|ico)(\?|$)", re.I)

# Paragraphs from the story container. Verse pages nest <p> per hemistich inside a
# div.w; prose pages put the paragraph text directly in div.w.
JS_EXTRACT = r"""
() => {
  const c = document.querySelector('div.pMC.showTranslation')
        || document.querySelector('div.pMC[data-pc]');
  if (!c) return [];
  const out = [];
  for (const w of c.querySelectorAll('div.w')) {
    const ps = w.querySelectorAll('p');
    if (ps.length) {
      for (const p of ps) {
        const t = (p.innerText || '').replace(/\s+/g,' ').trim();
        if (t) out.push(t);
      }
    } else {
      const t = (w.innerText || '').replace(/\s+/g,' ').trim();
      if (t) out.push(t);
    }
  }
  return out;
}
"""


class _StoryParser(html.parser.HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.container_depth = None
        self.paragraph_depth = None
        self.paragraph = []
        self.containers = []
        self.current = []

    def handle_starttag(self, tag, attrs):
        if tag != "div":
            return
        self.depth += 1
        classes = dict(attrs).get("class", "").split()
        if self.container_depth is None and "pMC" in classes:
            self.container_depth = self.depth
            self.current = []
        elif self.container_depth is not None and self.paragraph_depth is None and "w" in classes:
            self.paragraph_depth = self.depth
            self.paragraph = []

    def handle_data(self, data):
        if self.paragraph_depth is not None:
            self.paragraph.append(data)

    def handle_endtag(self, tag):
        if tag != "div":
            return
        if self.paragraph_depth == self.depth:
            text = re.sub(r"\s+", " ", "".join(self.paragraph)).strip()
            if text:
                self.current.append(text)
            self.paragraph_depth = None
        if self.container_depth == self.depth:
            self.containers.append(self.current)
            self.container_depth = None
        self.depth -= 1


def story_from_html(source):
    """Extract paragraphs from server-rendered Rekhta HTML without a browser."""
    parser = _StoryParser()
    parser.feed(source)
    candidates = [[paragraph for paragraph in container if keep(paragraph)]
                  for container in parser.containers]
    return max(candidates, key=len, default=[])


def fetch_story(url):
    req = urllib.request.Request(
        with_lang(url),
        headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "ur,en;q=0.5"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return story_from_html(response.read().decode("utf-8", "replace"))


def with_lang(u):
    return u + ("&" if "?" in u else "?") + "lang=ur"


def keep(l):
    if not l:
        return False
    n = len(URDU.findall(l))
    return n >= 8 and n / max(1, len(l)) >= 0.45


def slug_of(u):
    return u.rstrip("/").split("/")[-1].split("?")[0]


async def new_page(ctx):
    pg = await ctx.new_page()
    await pg.route(BLOCK, lambda r: asyncio.ensure_future(r.abort()))
    return pg


async def discover_kinds(pg, author):
    """Which content types this author actually has."""
    await pg.goto(with_lang(f"{BASE}/{SPACE}/{author}"), timeout=NAV_TIMEOUT,
                  wait_until="networkidle")
    hrefs = await pg.eval_on_selector_all(
        "a[href]", "els => els.map(e => e.getAttribute('href'))")
    found = []
    for h in hrefs or []:
        if not h:
            continue
        m = re.search(rf"/{SPACE}/{re.escape(author)}/([a-z0-9\-]+)$", h.split("?")[0])
        if m and m.group(1) not in found:
            found.append(m.group(1))
    return found


async def enumerate_kind(pg, author, kind, max_pages=40):
    want = re.compile(rf"^/{re.escape(kind)}/[^/?#]+$")
    urls, seen = [], set()
    for i in range(1, max_pages + 1):
        await pg.goto(with_lang(f"{BASE}/{SPACE}/{author}/{kind}?pageIndex={i}"),
                      timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
        hrefs = await pg.eval_on_selector_all(
            "a[href]", "els => els.map(e => e.getAttribute('href'))")
        fresh = []
        for h in hrefs or []:
            if not h:
                continue
            h = re.sub(r"^https?://(www\.)?rekhta\.org", "", h.split("?")[0].split("#")[0])
            if not want.match(h):
                continue
            full = BASE + h
            if full not in seen:
                seen.add(full)
                fresh.append(full)
        if not fresh:
            break
        urls += fresh
    return urls


async def story(pg, url):
    await pg.goto(with_lang(url), timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
    try:
        await pg.wait_for_selector("div.pMC.showTranslation div.w", timeout=SEL_TIMEOUT)
    except Exception:                                       # noqa: BLE001
        pass
    paras = [p for p in await pg.evaluate(JS_EXTRACT) if keep(p)]
    seen, out = set(), []
    for p in paras:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


async def main():
    cmd, author = sys.argv[1], sys.argv[2]
    if cmd == "page":
        paragraphs = fetch_story(author)
        if not paragraphs:
            print("no story paragraphs found", file=sys.stderr)
            return 1
        print("\n".join(paragraphs))
        return 0

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="ur-PK",
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        pg = await new_page(ctx)

        if cmd == "kinds":
            got = await discover_kinds(pg, author)
            print("\n".join(got))
            await browser.close()
            return

        if cmd == "index":
            kinds = ([sys.argv[3]] if len(sys.argv) > 3 and sys.argv[3] != "all"
                     else list(KINDS))
            for k in kinds:
                us = await enumerate_kind(pg, author, k)
                print(f"# {k}: {len(us)}", file=sys.stderr, flush=True)
                for u in us:
                    print(u)
            await browser.close()
            return

        outdir = sys.argv[3]
        kinds = ([sys.argv[4]] if len(sys.argv) > 4 and sys.argv[4] != "all"
                 else list(KINDS))
        os.makedirs(outdir, exist_ok=True)
        urls, seen = [], set()
        for k in kinds:
            us = await enumerate_kind(pg, author, k)
            print(f"# {k}: {len(us)}", flush=True)
            for u in us:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        await pg.close()

        def path_of(u):
            return os.path.join(outdir, slug_of(u) + ".txt")

        todo = [u for u in urls
                if not (os.path.exists(path_of(u)) and os.path.getsize(path_of(u)) > 0)]
        print(f"{len(urls)} items, {len(urls)-len(todo)} on disk, {len(todo)} to fetch, "
              f"{WORKERS} workers", flush=True)

        q = asyncio.Queue()
        for u in todo:
            q.put_nowait(u)
        stats = {"ok": 0, "fail": 0}

        async def worker():
            w = await new_page(ctx)
            while True:
                try:
                    u = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    paras = await story(w, u)
                except Exception as e:                       # noqa: BLE001
                    paras = []
                    print(f"  ERR {slug_of(u)}: {type(e).__name__}", flush=True)
                if len(paras) < 2:
                    stats["fail"] += 1
                    print(f"  FAIL {slug_of(u)}", flush=True)
                else:
                    with open(path_of(u), "w", encoding="utf-8") as f:
                        f.write("\n".join(paras) + "\n")
                    stats["ok"] += 1
                done = stats["ok"] + stats["fail"]
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} ok={stats['ok']} "
                          f"fail={stats['fail']}", flush=True)
                await asyncio.sleep(PAUSE)
            await w.close()

        await asyncio.gather(*[worker() for _ in range(WORKERS)])
        print(f"done: ok={stats['ok']} fail={stats['fail']}", flush=True)
        await browser.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
