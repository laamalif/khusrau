#!/usr/bin/env python3
"""Integrity check on the Ganjoor cache: completeness and non-emptiness.

Two failure modes worth catching separately, because the fetcher can report success
on both:

  gaps      -- a poem listed in cat_<id>.json with no file on disk. api.ganjoor.net
               answers 404 when it throttles rather than 429, so a poem that was
               merely rate-limited looks the same as one that does not exist. A
               "1 missing" line scrolls past in a long log; this makes it a hard fail.
  empties   -- a file that exists but has zero verses. Cached successfully, useless
               as reference, and invisible to any count-based check.

Exit status is non-zero if either is found, so this can gate the pipeline.
"""
import json
import os
import sys

from ganjoor import POETS, WORKS, _find, cat_poems


def check(cat_id):
    poems = cat_poems(cat_id)
    missing, empty, verses, lines = [], [], 0, 0
    for meta in poems:
        p = _find(f"poem_{meta['id']}.json", cat_id)
        if not os.path.exists(p):
            missing.append(meta["id"])
            continue
        rec = json.load(open(p, encoding="utf-8"))
        vs = [v for v in (rec.get("verses") or []) if v.strip()]
        if not vs:
            empty.append(meta["id"])
        verses += len(vs)
        lines += 1
    return {"listed": len(poems), "files": lines, "missing": missing,
            "empty": empty, "verses": verses}


def main():
    poets = sys.argv[1:] or sorted(POETS)
    bad = 0
    grand = 0
    for poet in poets:
        print(f"\n{poet}")
        tot_listed = tot_verses = 0
        for cat_id in POETS[poet]:
            r = check(cat_id)
            tot_listed += r["listed"]
            tot_verses += r["verses"]
            flag = ""
            if r["missing"]:
                flag += f"  MISSING {len(r['missing'])}: {r['missing'][:5]}"
                bad += len(r["missing"])
            if r["empty"]:
                flag += f"  EMPTY {len(r['empty'])}: {r['empty'][:5]}"
                bad += len(r["empty"])
            print(f"  {WORKS[cat_id][1]:22s} cat {cat_id}  "
                  f"listed {r['listed']:4d}  files {r['files']:4d}  "
                  f"verses {r['verses']:6d}{flag or '  ok'}")
        print(f"  {'TOTAL':22s}          listed {tot_listed:4d}"
              f"                verses {tot_verses:6d}")
        grand += tot_verses

    print(f"\ngrand total hemistichs: {grand}")
    if bad:
        print(f"FAIL: {bad} missing or empty poems", file=sys.stderr)
        return 1
    print("OK: every listed poem is present and non-empty")
    return 0


if __name__ == "__main__":
    sys.exit(main())
