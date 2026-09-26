#!/usr/bin/env python3
"""Rekhta scraper via Playwright. Runs inside the Playwright container on `vagrant`.

Why a browser is unavoidable here. Probing established (2026-09-04):
  - `?lang=ur` makes Rekhta render Urdu *metadata* server-side (title, topics, singer
    names) but NOT the poem body.
  - A poem page's HTML contains exactly one `div.pMC` container, and on the pages
    probed it held a promoted sher, in ROMAN, as per-word `<span data-m="CODE">`.
  - So the couplets are assembled client-side. No plain-HTTP shortcut exists.

Extraction targets Rekhta's own poem markup rather than visible-text heuristics:
    div.pMC  >  div.w[data-p]        one "w" per couplet
             >  div.c  >  p[data-l]  one p per hemistich, data-l = line number
Falling back to visible text only if that structure is absent, because a heuristic
pass silently swept in five navigation strings the first time it was tried on the
sister site (see fladdra.py) and that inflated CER by ~0.10 across the board.

Usage (inside container):
  rekhta_pw.py index <poet-slug>            -> slugs, one per line
  rekhta_pw.py fetch <poet-slug> <outdir>   -> restartable download
  rekhta_pw.py page <url>                   -> one poem to stdout
"""
import os
import re
import sys
import time

BASE = "https://www.rekhta.org"
PAUSE = float(os.environ.get("REKHTA_PAUSE", "1.2"))
NAV_TIMEOUT = int(os.environ.get("REKHTA_TIMEOUT", "45000"))
URDU = re.compile(r"[؀-ۿ]")

# Extract from Rekhta's poem markup. Returns [] if the structure is not present, so
# the caller can tell "no poem here" from "poem with zero Urdu".
JS_EXTRACT = r"""
() => {
  const out = [];
  const conts = document.querySelectorAll('div.pMC');
  for (const c of conts) {
    const couplets = c.querySelectorAll('div.w');
    for (const w of couplets) {
      for (const p of w.querySelectorAll('p')) {
        const t = (p.innerText || p.textContent || '').replace(/\s+/g, ' ').trim();
        if (t) out.push(t);
      }
    }
  }
  return out;
}
"""


def urdu_ratio(s):
    return len(URDU.findall(s)) / max(1, len(s))


def keep(line):
    """A hemistich: mostly Urdu, not a nav string, not a concatenated blob."""
    if not line or len(line) > 200:
        return False
    return len(URDU.findall(line)) >= 4 and urdu_ratio(line) >= 0.5


def with_lang(url):
    return url + ("&" if "?" in url else "?") + "lang=ur"


class Rekhta:
    def __init__(self, pw, proxy=None):
        args = {"headless": True}
        if proxy:
            args["proxy"] = {"server": proxy}
        self.browser = pw.chromium.launch(**args)
        self.ctx = self.browser.new_context(
            locale="ur-PK",
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        self.page = self.ctx.new_page()
        # Images and fonts are the bulk of the bytes and none of the text. Blocking
        # them cuts page time substantially; the poem is DOM text, not the webfont.
        self.page.route(re.compile(r"\.(png|jpe?g|gif|webp|svg|woff2?|ttf|mp4|mp3)$"),
                        lambda r: r.abort())

    def close(self):
        self.browser.close()

    def poem(self, url):
        self.page.goto(with_lang(url), timeout=NAV_TIMEOUT,
                       wait_until="domcontentloaded")
        try:
            self.page.wait_for_selector("div.pMC div.w p", timeout=15000)
        except Exception:                                   # noqa: BLE001
            pass
        lines = [l for l in self.page.evaluate(JS_EXTRACT) if keep(l)]
        seen, out = set(), []
        for l in lines:
            if l not in seen:
                seen.add(l)
                out.append(l)
        return out

    # Rekhta files a poet's work under several content types, and `/ghazals` is only
    # one of them. For Ghalib it lists exactly 233 there -- the published divan -- while
    # the verse actually needed to align Nuskha-e-Hamidiya and Nuskha-e-Sherani lives
    # under `unpublished-ghazal` and `qasiida`. Enumerating only `/ghazals` silently
    # misses the whole ghair-matbua layer, which is the point of those manuscripts.
    # `latiife` and `letters` are deliberately NOT here. They are prose: no <p> per
    # hemistich, so this extractor returns nothing and every item logs FAIL. Letters
    # belong to rekhta_prose.py, which reads a div.w as one paragraph. latiife
    # (anecdotes) are skipped entirely -- not reference text for any target.
    KINDS = ("ghazals", "unpublished-ghazal", "qasiida", "qita", "nazms",
             "rubaai", "masnavi", "salaam", "marsiya", "sehra", "mukhammas")
    _POEM_HREF = re.compile(
        r"/(ghazals|unpublished-ghazal|qasiida|qita|nazms|rubaai|masnavi"
        r"|salaam|marsiya|sehra|mukhammas)/[^/?#]+$")

    def poet_poems(self, poet, kind="ghazals", max_pages=60):
        urls, seen = [], set()
        for i in range(1, max_pages + 1):
            u = f"{BASE}/poets/{poet}/{kind}?pageIndex={i}"
            self.page.goto(with_lang(u), timeout=NAV_TIMEOUT,
                           wait_until="domcontentloaded")
            hrefs = self.page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.getAttribute('href'))")
            fresh = []
            for h in hrefs or []:
                if not h:
                    continue
                h = h.split("?")[0].split("#")[0]
                if not self._POEM_HREF.search(h):
                    continue
                full = h if h.startswith("http") else BASE + h
                if full not in seen:
                    seen.add(full)
                    fresh.append(full)
            if not fresh:
                break
            urls += fresh
            time.sleep(PAUSE)
        return urls

    # kept for callers that predate the kind argument
    def poet_ghazals(self, poet, max_pages=60):
        return self.poet_poems(poet, "ghazals", max_pages)


def slug_of(url):
    return url.rstrip("/").split("/")[-1].split("?")[0]


def main():
    from playwright.sync_api import sync_playwright
    cmd = sys.argv[1]
    proxy = os.environ.get("REKHTA_PROXY") or None
    with sync_playwright() as pw:
        rk = Rekhta(pw, proxy)
        try:
            if cmd == "page":
                print("\n".join(rk.poem(sys.argv[2])))
            elif cmd == "index":
                poet = sys.argv[2]
                kinds = ([sys.argv[3]] if len(sys.argv) > 3
                         and sys.argv[3] != "all" else list(rk.KINDS))
                for k in kinds:
                    us = rk.poet_poems(poet, k)
                    print(f"# {k}: {len(us)}", file=sys.stderr, flush=True)
                    for u in us:
                        print(u)
            elif cmd == "fetch":
                poet, outdir = sys.argv[2], sys.argv[3]
                kinds = ([sys.argv[4]] if len(sys.argv) > 4
                         and sys.argv[4] != "all" else list(rk.KINDS))
                os.makedirs(outdir, exist_ok=True)
                urls, seen = [], set()
                for k in kinds:
                    us = rk.poet_poems(poet, k)
                    print(f"# {k}: {len(us)}", flush=True)
                    for u in us:
                        if u not in seen:
                            seen.add(u)
                            urls.append(u)
                print(f"{len(urls)} ghazal urls", flush=True)
                ok = skip = fail = 0
                for i, u in enumerate(urls):
                    p = os.path.join(outdir, slug_of(u) + ".txt")
                    if os.path.exists(p) and os.path.getsize(p) > 0:
                        skip += 1
                        continue
                    try:
                        lines = rk.poem(u)
                    except Exception as e:                   # noqa: BLE001
                        lines = []
                        print(f"  ERR {slug_of(u)}: {type(e).__name__}", flush=True)
                    if len(lines) < 2:
                        fail += 1
                        print(f"  FAIL {slug_of(u)}", flush=True)
                    else:
                        with open(p, "w", encoding="utf-8") as f:
                            f.write("\n".join(lines) + "\n")
                        ok += 1
                    time.sleep(PAUSE)
                    if (i + 1) % 25 == 0:
                        print(f"  {i+1}/{len(urls)} ok={ok} skip={skip} fail={fail}",
                              flush=True)
                print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)
        finally:
            rk.close()


if __name__ == "__main__":
    main()
