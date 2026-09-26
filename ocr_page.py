#!/usr/bin/env python3
"""Read one page of Perso-Arabic verse with Gemini. Deliberately layout-naive.

The prompt does NOT reveal the divan column convention (§18.2: right column is
misra 1, left column is misra 2, paired row-wise). Telling the model would measure
the prompt, not the reader. Reading order is part of the task, and §18.2 showed the
reference can repair it afterwards anyway.

Uncertain-word markers are requested because §17.2 found self-reported uncertainty
is calibrated -- it predicts where the real errors are, so it is a routing signal
for the adjudication budget, not noise.

Usage: ocr_page.py <image> [more images...]   -> writes <image>.txt beside a .json
"""
import json
import os
import sys

from vertex import generate, image_part

SYSTEM = (
    "You transcribe pages of lithographed and letterpress Perso-Arabic verse "
    "(Persian and Urdu, nasta'liq or naskh). Output ONLY the transcription.\n"
    "Rules:\n"
    "- One hemistich (misra) per output line. Do not merge two hemistichs onto one line.\n"
    "- Preserve the original script and orthography exactly as printed. Do NOT modernise "
    "spelling, do NOT add or remove izafat, and do NOT normalise Urdu letterforms "
    "(ہ ے ں ٹ ڈ ڑ ھ) to Persian ones.\n"
    "- Transcribe every line of verse on the page, in the order a reader would read them.\n"
    "- Wrap a word you are unsure of as [word?]. Use [ILL] for an illegible word. "
    "Being explicit about doubt is more useful than guessing silently.\n"
    "- Put section headings on their own line prefixed with '## '.\n"
    "- Ignore the ruled border, running heads, page numbers and catchwords.\n"
    "- No translation, no transliteration, no commentary, no line numbers."
)

PROSE_SYSTEM = (
    "You transcribe pages of lithographed and letterpress Urdu prose in nasta'liq. "
    "Output ONLY the transcription.\n"
    "Rules:\n"
    "- Preserve the original script and orthography exactly as printed.\n"
    "- Preserve paragraph boundaries: output one printed paragraph per line.\n"
    "- Transcribe text continued from a previous page or onto the next page normally.\n"
    "- Wrap an uncertain word as [word?]. Use [ILL] for an illegible word.\n"
    "- Put section headings on their own line prefixed with '## '.\n"
    "- Ignore page numbers, running heads, ruled borders, and catchwords.\n"
    "- Do not translate, transliterate, summarize, modernize, or add commentary."
)

DICTIONARY_SYSTEM = (
    "You perform exact OCR of a modern printed Urdu literary dictionary in nasta'liq. "
    "Output ONLY the transcription.\n"
    "Rules:\n"
    "- The page has two columns. Read the RIGHT column completely from top to bottom, "
    "then the LEFT column completely from top to bottom. Never alternate between columns.\n"
    "- Insert one blank line at the column boundary. Do not add column labels.\n"
    "- Preserve every headword, definition, quotation, attribution, source citation, "
    "page reference, numeral, and punctuation mark in its original script.\n"
    "- Keep each dictionary headword and its definition together. Put each printed verse "
    "line on its own output line and do not merge separate verse lines.\n"
    "- Continue text across the column boundary exactly as printed. Do not invent text "
    "that is clipped or continued from another page.\n"
    "- Preserve the original spelling and letterforms. Do not modernize, translate, "
    "transliterate, explain, silently correct, or add Markdown.\n"
    "- Use [ILL] only for genuinely illegible text and wrap an uncertain reading as "
    "[word?].\n"
    "- Ignore only the running page number and blank margins."
)


def read_page(path, model=None, thinking=None, max_tokens=32768, mode="verse"):
    """max_tokens is deliberately large: on gemini-3.x the thinking budget is drawn
    from the same pool, so 8192 truncated a 22-line page (§19)."""
    systems = {
        "verse": SYSTEM,
        "prose": PROSE_SYSTEM,
        "dictionary": DICTIONARY_SYSTEM,
    }
    try:
        system = systems[mode]
    except KeyError:
        raise ValueError(f"unsupported OCR mode: {mode}") from None
    return generate([image_part(path), {"text": "Transcribe this page."}],
                    model=model, system=system, max_tokens=max_tokens,
                    temperature=0, thinking=thinking)


def artifact_paths(path, suffix=""):
    base = os.path.splitext(path)[0] + suffix
    return base + ".txt", base + ".partial.txt", base + ".response.json"


def write_response(path, response):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main():
    if len(sys.argv) < 2:
        print("usage: ocr_page.py <image> [more images...]", file=sys.stderr)
        return 2
    model = os.environ.get("HTR_MODEL")
    thinking = os.environ.get("HTR_THINKING")
    mode = os.environ.get("HTR_MODE", "verse")
    suffix = os.environ.get("HTR_SUFFIX", "")
    failed = 0
    for path in sys.argv[1:]:
        r = read_page(path, model, thinking, mode=mode)
        r["request"] = {"model": model, "thinking": thinking, "mode": mode}
        out, partial, response = artifact_paths(path, suffix)
        write_response(response, r)
        if "error" in r:
            print(f"{path}: ERROR {r['error'][:200]}", file=sys.stderr)
            failed += 1
            continue
        text = r.get("text", "").strip()
        complete = r.get("finish") == "STOP"
        target = out if complete else partial
        if text:
            with open(target, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        if not complete:
            failed += 1
        u = r.get("usage", {})
        print(f"{path} -> {target if text else response}  finish={r.get('finish')}  "
              f"in={u.get('promptTokenCount')} out={u.get('candidatesTokenCount')}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
