#!/usr/bin/env python3
"""Scrape ghalib.fladdra.com, and measure whether its text can be trusted.

Why this source at all. Rekhta does not server-render poem bodies: `?lang=ur` yields
Urdu *metadata* only (title, topics, singers), and the one `pMC` container on a poem
page holds a promoted sher with per-word `data-m` codes, in roman. Scraping the body
would need a headless browser, and none is installed here. fladdra serves the Urdu
directly in plain HTML.

fladdra lists **290** Ghalib poems against the 233 in scratch/classic, and the 60 extra
slugs are qasidas and occasional verse (`ai-shahnshaah-e-aasmaan-aurang`,
`bhejii-hai-jo-mujh-ko-shaah-e-jamjaah-ne-daal`) -- i.e. the ghair-matbua layer needed
to align Nuskha-e-Hamidiya and Nuskha-e-Sherani, which carry verse Ghalib kept out of
his published divan.

The trust question is answerable without Rekhta. The local 233 files use Rekhta slugs
and are Rekhta-derived, so comparing fladdra against them on the overlap is a
transitive check on fladdra vs Rekhta. `verify` does that and reports CER.

Usage:
  fladdra.py index                    # list slugs
  fladdra.py fetch <outdir>           # download all, restartable
  fladdra.py verify <localdir> [n]    # CER of fladdra vs local Rekhta-derived files
"""
import html
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://ghalib.fladdra.com"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
PAUSE = float(os.environ.get("FLADDRA_PAUSE", "0.8"))
PROXY = os.environ.get("SCRAPE_PROXY", "")
URDU = re.compile(r"[؀-ۿ]")

if PROXY and PROXY.lower() not in ("", "none", "off"):
    _p = PROXY if "://" in PROXY else f"http://{PROXY}"
    _opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": _p, "https": _p}))
else:
    _opener = urllib.request.build_opener()


def get(url, retries=4):
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with _opener.open(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(min(2 ** a, 15))
    return None


def index():
    h = get(f"{BASE}/ghazals/")
    if not h:
        raise SystemExit("index fetch failed")
    return list(dict.fromkeys(re.findall(r'href="/ghazals/([^"/]+)/"', h)))


# Page furniture, not poem text. Verified by diffing against the local Rekhta-derived
# files: these five navigation strings were the ONLY difference on every ghazal
# checked, which is what established that fladdra's poem text is identical (see
# `verify`). Filtering by marker glyph rather than exact string so wording changes on
# the site do not silently reintroduce them.
_CHROME = {"اردو", "فہرست", "غزل", "غزل کاپی", "انتخابِ غالبؔ", "مرزا غالبؔ",
           "مرزا غالب", "ہندی", "رومن", "دیوان", "تمام غزلیں",
           "پچھلی غزل", "اگلی غزل"}
_CHROME_MARK = ("→", "←", "✓", "·", "•")


def poem_lines(h):
    h = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", h, flags=re.S | re.I)
    h = re.sub(r"<br\s*/?>", "\n", h, flags=re.I)
    h = re.sub(r"</(p|div|li|h[1-6]|tr)>", "\n", h, flags=re.I)
    h = re.sub(r"<[^>]+>", "\n", h)
    out, seen = [], set()
    for l in (html.unescape(x).strip() for x in h.split("\n")):
        if not l or l in _CHROME or "|" in l:
            continue
        if any(m in l for m in _CHROME_MARK):
            continue
        n = len(URDU.findall(l))
        if n < 6 or n / max(1, len(l)) < 0.5 or len(l) > 200:
            continue
        if l in seen:                 # the matla repeats as the page heading
            continue
        seen.add(l)
        out.append(l)
    return out


def fetch(outdir):
    os.makedirs(outdir, exist_ok=True)
    slugs = index()
    print(f"{len(slugs)} slugs", flush=True)
    ok = skip = fail = 0
    for i, s in enumerate(slugs):
        p = os.path.join(outdir, s + ".txt")
        if os.path.exists(p) and os.path.getsize(p) > 0:
            skip += 1
            continue
        h = get(f"{BASE}/ghazals/{s}/")
        lines = poem_lines(h) if h else []
        if len(lines) < 2:
            fail += 1
            print(f"  FAIL {s}", flush=True)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            ok += 1
        time.sleep(PAUSE)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(slugs)} ok={ok} skip={skip} fail={fail}", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)


def verify(localdir, limit=None):
    from urdu_norm import cer, norm_match
    local = {}
    for f in os.listdir(localdir):
        if "-mirza-ghalib-ghazals" in f and f.endswith(".txt"):
            local[f.replace("-mirza-ghalib-ghazals.txt", "")] = os.path.join(localdir, f)
    slugs = [s for s in index() if s in local]
    if limit:
        slugs = slugs[:limit]
    print(f"overlap with local: {len(slugs)} ghazals\n")
    rows = []
    for s in slugs:
        h = get(f"{BASE}/ghazals/{s}/")
        fl = poem_lines(h) if h else []
        lc = [l.strip() for l in open(local[s], encoding="utf-8") if l.strip()]
        a = " ".join(t for l in fl for t in norm_match(l))
        b = " ".join(t for l in lc for t in norm_match(l))
        rows.append((s, len(fl), len(lc), cer(a, b)))
        time.sleep(PAUSE)
    rows.sort(key=lambda r: -r[3])
    print(f"{'slug':52s} {'fl':>4s} {'loc':>4s} {'CER':>7s}")
    for s, nf, nl, c in rows:
        print(f"{s[:52]:52s} {nf:4d} {nl:4d} {c:7.4f}")
    cs = sorted(r[3] for r in rows)
    ident = sum(1 for r in rows if r[3] == 0.0)
    print(f"\nidentical: {ident}/{len(rows)}   "
          f"median CER {cs[len(cs)//2]:.4f}   max {cs[-1]:.4f}")
    linecount_mismatch = [r for r in rows if r[1] != r[2]]
    print(f"line-count mismatches: {len(linecount_mismatch)}/{len(rows)}")


def main():
    cmd = sys.argv[1]
    if cmd == "index":
        for s in index():
            print(s)
    elif cmd == "fetch":
        fetch(sys.argv[2])
    elif cmd == "verify":
        verify(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)


if __name__ == "__main__":
    main()
