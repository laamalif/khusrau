#!/usr/bin/env python3
"""Ganjoor reference side: fetch a poet's category, cache it, index it by hemistich.

Caching to disk is not an optimisation, it is a requirement. api.ganjoor.net rate
limits hard and returns **HTTP 404 rather than 429** when it does -- during this
project every poet id including ones that had just succeeded started 404ing. A 404
is therefore ambiguous between "no such poem" and "slow down", so the fetcher
retries with backoff and never overwrites a good cache entry with a failure.

Usage:
  ganjoor.py fetch <catId>     # e.g. 2572 = Ghalib Dehlavi, Persian ghazals (334)
  ganjoor.py index <catId>     # report index stats
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict

from urdu_norm import norm_match

API = "https://api.ganjoor.net/api/ganjoor"
CACHE = os.environ.get("GANJOOR_CACHE",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "work/ganjoor"))
PAUSE = float(os.environ.get("GANJOOR_PAUSE", "0.6"))
PROXY = os.environ.get("SCRAPE_PROXY", "")

# Scoped to this module's opener rather than exported as http_proxy, so the Vertex
# client keeps its direct path -- the service-account token exchange should not be
# routed through a scraping proxy.
if PROXY and PROXY.lower() not in ("", "none", "off"):
    _p = PROXY if "://" in PROXY else f"http://{PROXY}"
    _opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": _p, "https": _p}))
else:
    _opener = urllib.request.build_opener()

# Cache layout is <poet>/<work>/ rather than flat. A flat cache mixed Ghalib's Persian
# ghazals with Iqbal's seven Persian works, so you could not tell what had been pulled;
# and each Iqbal work is a separate book that gets aligned against a separate scan
# (asrarorumuz, zaburiajam, javidnamaaziqbal, armughanihijaz, payamimashriq), so keeping
# them apart is what lets a target be matched against just its own reference.
WORKS = {
    # Ghalib Dehlavi, Ganjoor poet 123
    2571: ("ghalib", "divan-e-ashaar"),
    2572: ("ghalib", "ghazaliyat"),          # 334 -- pairs with ghazaliyatifarsi
    2573: ("ghalib", "rubaiyat"),
    # Iqbal Lahori, Ganjoor poet 42
    1357: ("iqbal", "_root"),
    1358: ("iqbal", "asrar-e-khudi"),        # pairs with asrarorumuz
    1359: ("iqbal", "rumuz-e-bekhudi"),      #   "        "
    1360: ("iqbal", "payam-e-mashriq"),      # pairs with payamimashriq
    1361: ("iqbal", "cat-1361"),             # not in the poet listing; kept for safety
    1362: ("iqbal", "zabur-e-ajam"),         # pairs with zaburiajam
    1363: ("iqbal", "javid-nama"),           # pairs with javidnamaaziqbal
    1364: ("iqbal", "pas-che-bayad-kard"),
    1365: ("iqbal", "armughan-e-hijaz"),     # pairs with armughanihijaz
}
POETS = {"ghalib": [2572, 2573],
         "iqbal": [1358, 1359, 1360, 1362, 1363, 1364, 1365]}


def cache_dir(cat_id):
    poet, work = WORKS.get(cat_id, ("_other", f"cat_{cat_id}"))
    d = os.path.join(CACHE, poet, work)
    os.makedirs(d, exist_ok=True)
    return d


def _find(name, cat_id):
    """Prefer the per-poet path; fall back to the legacy flat path."""
    p = os.path.join(cache_dir(cat_id), name)
    if os.path.exists(p):
        return p
    legacy = os.path.join(CACHE, name)
    return legacy if os.path.exists(legacy) else p


def _get(url, retries=6):
    for a in range(retries):
        try:
            with _opener.open(
                    urllib.request.Request(url, headers={"accept": "application/json"}),
                    timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            # 404 here is usually throttling, not absence. Back off and retry.
            if e.code not in (404, 429, 500, 502, 503):
                raise
        except Exception:                                   # noqa: BLE001
            pass
        time.sleep(min(2 ** a, 30))
    return None


def cat_poems(cat_id):
    """List of {id,title} for a category, cached."""
    p = _find(f"cat_{cat_id}.json", cat_id)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    d = _get(f"{API}/cat/{cat_id}?poems=true")
    if not d:
        raise SystemExit(f"could not fetch cat {cat_id}")
    poems = [{"id": x["id"], "title": x.get("title")}
             for x in (d.get("cat", {}).get("poems") or [])]
    json.dump(poems, open(os.path.join(cache_dir(cat_id), f"cat_{cat_id}.json"),
                         "w", encoding="utf-8"), ensure_ascii=False)
    return poems


def poem(pid, cat_id=None):
    """{id,title,verses:[str]} for one poem, cached under its poet's directory."""
    p = _find(f"poem_{pid}.json", cat_id)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    d = _get(f"{API}/poem/{pid}")
    if not d:
        return None
    rec = {"id": pid, "title": d.get("title"), "fullTitle": d.get("fullTitle"),
           "verses": [v.get("text", "") for v in (d.get("verses") or [])]}
    out = os.path.join(cache_dir(cat_id), f"poem_{pid}.json") if cat_id is not None \
        else os.path.join(CACHE, f"poem_{pid}.json")
    json.dump(rec, open(out, "w", encoding="utf-8"), ensure_ascii=False)
    return rec


def migrate():
    """Move a legacy flat cache into per-poet directories. Idempotent."""
    import shutil
    moved = 0
    for name in sorted(os.listdir(CACHE)):
        if not name.startswith("cat_") or not name.endswith(".json"):
            continue
        cat_id = int(name[4:-5])
        src = os.path.join(CACHE, name)
        if not os.path.isfile(src):
            continue
        dst_dir = cache_dir(cat_id)
        poems = json.load(open(src, encoding="utf-8"))
        for meta in poems:
            f = f"poem_{meta['id']}.json"
            a, b = os.path.join(CACHE, f), os.path.join(dst_dir, f)
            if os.path.isfile(a) and not os.path.exists(b):
                shutil.move(a, b)
                moved += 1
        b = os.path.join(dst_dir, name)
        if not os.path.exists(b):
            shutil.move(src, b)
        print(f"  {name} -> {os.path.relpath(dst_dir, CACHE)}/  ({len(poems)} poems)")
    left = [f for f in os.listdir(CACHE)
            if f.startswith("poem_") and os.path.isfile(os.path.join(CACHE, f))]
    print(f"moved {moved} poem files; {len(left)} unclaimed poem files remain")


def fetch_all(cat_id):
    poems = cat_poems(cat_id)
    print(f"cat {cat_id}: {len(poems)} poems", flush=True)
    got = miss = 0
    for i, meta in enumerate(poems):
        if os.path.exists(_find(f"poem_{meta['id']}.json", cat_id)):
            got += 1
            continue
        r = poem(meta["id"], cat_id)
        if r:
            got += 1
        else:
            miss += 1
            print(f"  MISS {meta['id']} {meta.get('title')}", flush=True)
        time.sleep(PAUSE)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(poems)}  cached={got} missing={miss}", flush=True)
    print(f"done: {got} cached, {miss} missing", flush=True)
    return got, miss


class Index:
    """Hemistich index over a set of Ganjoor poems.

    Mirrors selftest_align.py's HemistichVote, which §5 measured at 98.3% recall /
    2.5% FPR: each query line finds its own best partner by character-trigram
    Jaccard and votes for that partner's poem.
    """

    DF_CAP = 3000

    def __init__(self, poems):
        self.poems = {p["id"]: p for p in poems}
        self.lines = []          # (poem_id, verse_index, trigrams, text)
        self.index = defaultdict(list)
        for p in poems:
            for j, v in enumerate(p["verses"]):
                toks = norm_match(v)
                if not toks:
                    continue
                tg = self.trigrams(" ".join(toks))
                if not tg:
                    continue
                lid = len(self.lines)
                self.lines.append((p["id"], j, tg, v))
                for t in tg:
                    self.index[t].append(lid)
        self.hot = {t for t, v in self.index.items() if len(v) > self.DF_CAP}

    @staticmethod
    def trigrams(s):
        s = s.replace(" ", "_")
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def match_line(self, text, min_jaccard=0.30):
        """Best (poem_id, verse_index, jaccard, ref_text) for one hypothesis line."""
        toks = norm_match(text)
        if not toks:
            return None
        qt = self.trigrams(" ".join(toks)) - self.hot
        if not qt:
            return None
        overlap = defaultdict(int)
        for t in qt:
            for lid in self.index.get(t, ()):
                overlap[lid] += 1
        best = None
        for lid, ov in overlap.items():
            pid, j, tg, raw = self.lines[lid]
            jac = ov / (len(qt) + len(tg) - ov)
            if best is None or (-jac, pid, j) < (-best[2], best[0], best[1]):
                best = (pid, j, jac, raw)
        if best and best[2] >= min_jaccard:
            return best
        return None


def load_index(*cat_ids, allow_partial=False):
    """Index one or more categories together.

    Multiple ids matter for Iqbal: asrarorumuz is one scan containing two Ganjoor
    works (اسرار خودی 1358 and رموز بیخودی 1359), so its reference index has to span
    both or half the pages will find no match.
    """
    poems = []
    problems = []
    for cat_id in cat_ids:
        for meta in cat_poems(cat_id):
            p = _find(f"poem_{meta['id']}.json", cat_id)
            if not os.path.exists(p):
                problems.append(f"cat {cat_id}: missing poem {meta['id']}")
                continue
            with open(p, encoding="utf-8") as f:
                rec = json.load(f)
            if not any(v.strip() for v in (rec.get("verses") or [])):
                problems.append(f"cat {cat_id}: empty poem {meta['id']}")
                continue
            poems.append(rec)
    if problems and not allow_partial:
        detail = "\n".join(f"  {p}" for p in problems[:20])
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        raise RuntimeError(
            "Ganjoor cache is incomplete; run `ganjoor.py fetch` and "
            "`verify_ganjoor.py` before indexing:\n" + detail + more)
    return Index(poems)


def main():
    cmd = sys.argv[1]
    if cmd == "migrate":
        migrate()
        return
    if cmd == "tree":
        for poet in sorted(POETS):
            print(poet)
            for c in POETS[poet]:
                d = cache_dir(c)
                n = len([f for f in os.listdir(d) if f.startswith("poem_")])
                print(f"  {WORKS[c][1]:22s} cat {c}  {n:4d} poems")
        return
    if cmd == "poet":
        name = sys.argv[2]
        for c in POETS[name]:
            fetch_all(c)
        return
    cat_ids = [int(x) for x in sys.argv[2:]]
    if cmd == "fetch":
        for c in cat_ids:
            fetch_all(c)
    elif cmd == "index":
        ix = load_index(*cat_ids)
        print(f"poems {len(ix.poems)} | hemistichs {len(ix.lines)} | "
              f"trigrams {len(ix.index)} | hot(skipped) {len(ix.hot)}")


if __name__ == "__main__":
    main()
