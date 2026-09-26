#!/usr/bin/env python3
"""Rekhta scraper for the Urdu reference side.

Two things make this work, both found by probing (2026-09-04):

  1. The default page is client-rendered -- plain HTTP returns a romanised title and
     zero Urdu codepoints, plus a `canvasIdDownLoad` div and a Mehr Nastaliq webfont.
     Appending **`?lang=ur`** makes the server render the Urdu text into the HTML.
     No headless browser is needed. `?lang=1` does not work; the cookie does not
     either. It is the query parameter.
  2. `robots.txt` disallows only `/home/`, `/fonts/`, `/uploads/`. Poet and ghazal
     pages are not disallowed. Requests are paced and identify a real UA; keep it
     that way.

Couplets are emitted one hemistich per line, matching the shape the rest of this
project expects (corpus.py, ganjoor.py) and the shape of the existing
scratch/classic files.

Usage:
  rekhta.py page <url>                 # one poem -> stdout
  rekhta.py index <poet-slug> [...]    # enumerate a poet's ghazal URLs
  rekhta.py fetch <poet-slug> <outdir> # enumerate + download, restartable
"""
import html
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BASE = "https://www.rekhta.org"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
PAUSE = float(os.environ.get("REKHTA_PAUSE", "1.5"))
PROXY = os.environ.get("REKHTA_PROXY", "")
URDU = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")

# Route through the proxy rather than exporting http_proxy, so only this scraper is
# affected -- the Vertex and Ganjoor clients keep their direct paths.
if PROXY and PROXY.lower() not in ("", "none", "off"):
    _p = PROXY if "://" in PROXY else f"http://{PROXY}"
    _opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": _p, "https": _p}))
else:
    _opener = urllib.request.build_opener()


def get(url, retries=4):
    # ?lang=ur is what forces server-side Urdu rendering. Add it, don't clobber.
    parts = urllib.parse.urlsplit(url)
    q = dict(urllib.parse.parse_qsl(parts.query))
    q["lang"] = "ur"
    url = urllib.parse.urlunsplit(
        parts._replace(query=urllib.parse.urlencode(q)))
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": UA,
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ur,en;q=0.5"})
            with _opener.open(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(min(2 ** a, 20))
    return None


def _visible_lines(h):
    h = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", h,
               flags=re.S | re.I)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"</(p|div|li|h[1-6])>", "\n", h, flags=re.I)
    h = re.sub(r"<[^>]+>", "\n", h)
    return [html.unescape(x).strip() for x in h.split("\n")]


# Furniture that appears on every poem page and is not part of the poem.
_CHROME = {
    "اردو", "فہرست", "غزل", "غزل کاپی", "ہندی", "انگریزی", "رومن",
    "شیئر", "مزید", "دیوان", "ریختہ", "صفحہ اول",
}


def poem_lines(h):
    """Hemistichs of the poem, in order, one per line."""
    out, seen = [], set()
    for l in _visible_lines(h):
        if not l or l in _CHROME:
            continue
        # a hemistich is mostly Urdu and reasonably long
        n_ur = len(URDU.findall(l))
        if n_ur < 6 or n_ur / max(1, len(l)) < 0.5:
            continue
        if len(l) > 200:            # concatenated blob, not a hemistich
            continue
        if l in seen:               # Rekhta repeats the matla in the header
            continue
        seen.add(l)
        out.append(l)
    return out


def poet_ghazal_urls(poet_slug, max_pages=60):
    """Every ghazal URL for a poet, following Rekhta's paged index."""
    urls, page = [], 1
    seen = set()
    while page <= max_pages:
        u = f"{BASE}/poets/{poet_slug}/ghazals?pageIndex={page}"
        h = get(u)
        if not h:
            break
        # ORIGIN-NORMALISE FIRST. Rekhta now emits ABSOLUTE hrefs on listing pages, so the
        # old root-relative pattern matched nothing and this returned 0 poems for every poet
        # -- a silent empty result, which is worse than a crash. Found 2026-09-06; see
        # rekhta_poet.py, which supersedes this for verse.
        found = [re.sub(r'^https?://(www\.)?rekhta\.org', '', x.split('?')[0].split('#')[0])
                 for x in re.findall(r'href="([^"]+)"', h)]
        found = [x for x in found if re.match(r'^/ghazals/[^/]+$', x)]
        fresh = [f"{BASE}{x}" for x in dict.fromkeys(found)
                 if f"{BASE}{x}" not in seen]
        if not fresh:
            break
        for x in fresh:
            seen.add(x)
        urls += fresh
        page += 1
        time.sleep(PAUSE)
    return urls


def slug_of(url):
    return urllib.parse.urlsplit(url).path.rstrip("/").split("/")[-1]


def fetch_poet(poet, outdir):
    os.makedirs(outdir, exist_ok=True)
    urls = poet_ghazal_urls(poet)
    print(f"{len(urls)} ghazal urls", flush=True)
    if not urls:
        print("FAIL: no ghazal URLs found", flush=True)
        return 0, 0, 1
    ok = skip = fail = 0
    for i, u in enumerate(urls):
        p = os.path.join(outdir, slug_of(u) + ".txt")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            skip += 1
            continue
        h = get(u)
        lines = poem_lines(h) if h else []
        if len(lines) < 2:
            fail += 1
            print(f"  FAIL {slug_of(u)}", flush=True)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            ok += 1
        time.sleep(PAUSE)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(urls)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)
    return ok, skip, fail


def main():
    cmd = sys.argv[1]
    if cmd == "page":
        h = get(sys.argv[2])
        if not h:
            raise SystemExit("fetch failed")
        print("\n".join(poem_lines(h)))
    elif cmd == "index":
        for u in poet_ghazal_urls(sys.argv[2]):
            print(u)
    elif cmd == "fetch":
        poet, outdir = sys.argv[2], sys.argv[3]
        _, _, fail = fetch_poet(poet, outdir)
        return 1 if fail else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
