#!/usr/bin/env python3
"""Concurrent Rekhta fetcher. Same extraction as rekhta_pw.py, N pages at once.

Sequential measured ~20 s/poem (Rekhta pages are heavy even with images and fonts
blocked), which is ~2.8 h for Ghalib's 566 poems. The work is entirely network-bound
so concurrency is the whole fix. Kept modest (4) rather than maximal: this is someone
else's site and the earlier robots.txt check only established that these paths are
not disallowed, not that hammering them is welcome.

Restartable by design -- each poem is written the moment it returns and an existing
non-empty .txt is never refetched, so this can be killed and resumed freely. That
matters because the sequential run had already banked ~30 files.

Usage (inside container):
  rekhta_pw_fast.py <poet> <outdir> [kind|all]     [REKHTA_WORKERS=4]
"""
import asyncio
import os
import re
import sys

BASE = "https://www.rekhta.org"
WORKERS = int(os.environ.get("REKHTA_WORKERS", "4"))
NAV_TIMEOUT = int(os.environ.get("REKHTA_TIMEOUT", "45000"))
SEL_TIMEOUT = int(os.environ.get("REKHTA_SEL_TIMEOUT", "12000"))
PAUSE = float(os.environ.get("REKHTA_PAUSE", "0.4"))
URDU = re.compile(r"[؀-ۿ]")

# `latiife` and `letters` are deliberately excluded: both are prose, so this verse
# extractor finds no <p> per hemistich and logs FAIL on every item. Letters are fetched
# by rekhta_prose.py; latiife (anecdotes) are skipped entirely.
KINDS = ("ghazals", "unpublished-ghazal", "qasiida", "qita", "nazms", "rubaai",
         "masnavi", "salaam", "marsiya", "sehra", "mukhammas")
POEM_HREF = re.compile(
    r"/(" + "|".join(KINDS) + r")/[^/?#]+$")

JS_EXTRACT = r"""
() => {
  const out = [];
  for (const c of document.querySelectorAll('div.pMC')) {
    for (const w of c.querySelectorAll('div.w')) {
      for (const p of w.querySelectorAll('p')) {
        const t = (p.innerText || p.textContent || '').replace(/\s+/g,' ').trim();
        if (t) out.push(t);
      }
    }
  }
  return out;
}
"""

BLOCK = re.compile(r"\.(png|jpe?g|gif|webp|svg|woff2?|ttf|eot|mp4|mp3|ico)(\?|$)", re.I)


def with_lang(u):
    return u + ("&" if "?" in u else "?") + "lang=ur"


def keep(l):
    if not l or len(l) > 200:
        return False
    n = len(URDU.findall(l))
    return n >= 4 and n / max(1, len(l)) >= 0.5


def slug_of(u):
    return u.rstrip("/").split("/")[-1].split("?")[0]


async def new_page(ctx):
    pg = await ctx.new_page()
    await pg.route(BLOCK, lambda r: asyncio.ensure_future(r.abort()))
    return pg


async def poem(pg, url):
    await pg.goto(with_lang(url), timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
    try:
        await pg.wait_for_selector("div.pMC div.w p", timeout=SEL_TIMEOUT)
    except Exception:                                       # noqa: BLE001
        pass
    lines = [l for l in await pg.evaluate(JS_EXTRACT) if keep(l)]
    seen, out = set(), []
    for l in lines:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out


async def enumerate_kind(pg, poet, kind, max_pages=60):
    """URLs of THIS kind only.

    Must be restricted to the requested kind. Matching any kind in POEM_HREF picked up
    the sidebar cross-links every listing page carries, which made /nazms and /masnavi
    report 31 items each when Ghalib appears to have none -- the 31 were links to
    ghazals. The overall total stayed correct only because main() dedups across kinds,
    so the bug was invisible in the total and wrong in every per-kind number.
    """
    want = re.compile(rf"^/{re.escape(kind)}/[^/?#]+$")
    urls, seen = [], set()
    for i in range(1, max_pages + 1):
        await pg.goto(with_lang(f"{BASE}/poets/{poet}/{kind}?pageIndex={i}"),
                      timeout=NAV_TIMEOUT, wait_until="domcontentloaded")
        hrefs = await pg.eval_on_selector_all(
            "a[href]", "els => els.map(e => e.getAttribute('href'))")
        fresh = []
        for h in hrefs or []:
            if not h:
                continue
            h = re.sub(r"^https?://(www\.)?rekhta\.org", "",
                       h.split("?")[0].split("#")[0])
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


async def main():
    from playwright.async_api import async_playwright
    poet, outdir = sys.argv[1], sys.argv[2]
    kinds = ([sys.argv[3]] if len(sys.argv) > 3 and sys.argv[3] != "all"
             else list(KINDS))
    os.makedirs(outdir, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        ctx = await browser.new_context(
            locale="ur-PK",
            user_agent=("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        idx = await new_page(ctx)
        urls, seen = [], set()
        for k in kinds:
            us = await enumerate_kind(idx, poet, k)
            print(f"# {k}: {len(us)}", flush=True)
            for u in us:
                if u not in seen:
                    seen.add(u)
                    urls.append(u)
        await idx.close()

        todo = [u for u in urls
                if not (os.path.exists(os.path.join(outdir, slug_of(u) + ".txt"))
                        and os.path.getsize(os.path.join(outdir, slug_of(u) + ".txt")) > 0)]
        print(f"{len(urls)} urls, {len(urls)-len(todo)} already on disk, "
              f"{len(todo)} to fetch, {WORKERS} workers", flush=True)

        q = asyncio.Queue()
        for u in todo:
            q.put_nowait(u)
        stats = {"ok": 0, "fail": 0}

        async def worker(n):
            pg = await new_page(ctx)
            while True:
                try:
                    u = q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                try:
                    lines = await poem(pg, u)
                except Exception as e:                       # noqa: BLE001
                    lines = []
                    print(f"  ERR {slug_of(u)}: {type(e).__name__}", flush=True)
                if len(lines) < 2:
                    stats["fail"] += 1
                    print(f"  FAIL {slug_of(u)}", flush=True)
                else:
                    with open(os.path.join(outdir, slug_of(u) + ".txt"),
                              "w", encoding="utf-8") as f:
                        f.write("\n".join(lines) + "\n")
                    stats["ok"] += 1
                done = stats["ok"] + stats["fail"]
                if done % 25 == 0:
                    print(f"  {done}/{len(todo)} ok={stats['ok']} "
                          f"fail={stats['fail']}", flush=True)
                await asyncio.sleep(PAUSE)
            await pg.close()

        await asyncio.gather(*[worker(i) for i in range(WORKERS)])
        print(f"done: ok={stats['ok']} fail={stats['fail']}", flush=True)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
