#!/usr/bin/env python3
"""Frozen, content-hashed normalization profiles for Perso-Arabic text.

TWO PROFILES, PERMANENTLY SEPARATE, WITH OPPOSITE RISK ASYMMETRY.

  retrieval-v1   Candidate discovery. Over-folding is cheap: an extra candidate is
                 rejected downstream. Folds phonemic distinctions on purpose.
                 A false merge here costs nothing.

  apparatus-v1   Textual identity for a critical apparatus. A false merge deletes
                 textual history irrecoverably, so conservative is the default.
                 NEVER folds a distinction that could carry textual, lexical,
                 morphological or editorial information.

`apparatus-v1` is NOT "better normalization" -- it is a different operation for a
different scholarly purpose. Ten pairs that retrieval-v1 deliberately merges
(ہے/ہی, آب/اب, کھا/کہا, بھار/بہار, ٹال/تال, ڈال/دال, گڑ/گر, پھر/پہر, میں/مین,
کھانا/کہانا) are distinct words. See normalization_tests/must_not_merge.jsonl.

WHY THE APPARATUS PROFILE IS A LADDER, NOT A FUNCTION.
Hamid Ali Khan's own footnote states that he placed the اعراب on اِس / اُس / اِن / اُن
with deliberate care. Vowel marking is therefore *editorial evidence* in that
edition -- folding it away erases an editorial decision. But Rekhta marks no
harakat at all, so refusing to fold it means nothing ever compares equal. Neither
horn is acceptable, so the profile does not choose: it reports the SHALLOWEST LEVEL
at which two readings agree and erases nothing.

    L1 encoding       pure codepoint/presentation noise. Never informative.
    L2 orthographic   editorial sigla and press conventions: takhallus mark,
                      hamza-e-izafat carriers, hamza carriers, typographic quotes,
                      punctuation.
    L3 vocalisation   harakat. Editorially significant, hence its own rung.
    L4 segmentation   word division. Typesetting, not orthography.

`agreement_level(a, b)` returns "encoding" | "orthographic" | "vocalisation" |
"segmentation" | None. None means the readings differ in substance -- the only case
that is ambiguous between reader error and a genuine variant.

CONTENT HASHING. Every profile declares its rules as data (`SPEC`) and the executor
reads only from that data, so the hash covers the rules themselves and is unchanged
by comments or refactoring. Record `profile_hash` beside every normalized value;
when apparatus-v2 arrives you can explain exactly why a variant count moved.
"""
import hashlib
import json
import re
import unicodedata

# ---------------------------------------------------------------- rule data

# Our own transcription markup, stripped before anything else.
_MARKUP = {
    "leading_tag": r"^\s*\[[^\]]{0,40}\]\s*",          # [Urdu] / [Mixed: ...]
    "placeholder": r"\[(?:ILL|BLANK PAGE|PLATE|ORNAMENT)\]",
    "uncertain": r"\[([^\]]*?)\??\]",                   # [word?] -> word
}

# L1: Arabic/presentation codepoints that have an unambiguous Urdu equivalent.
# Nothing here can distinguish two words.
_L1_MAP = {
    "ي": "ی", "ى": "ی", "ﻯ": "ی", "ﻰ": "ی",
    "ك": "ک", "ﻙ": "ک", "ﻚ": "ک",
    "ه": "ہ", "ة": "ہ", "ۀ": "ہ", "ﻩ": "ہ", "ﮪ": "ہ",
    "ﮭ": "ھ", "ﻻ": "لا", "ﷲ": "اللہ",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "ﺍ": "ا",
}
_L1_DELETE = (
    "ـ"      # tatweel / kashida
    "‍"      # ZWJ
    "‎‏"  # LRM / RLM
    "​"      # ZWSP
    "­"      # soft hyphen
    "﻿"      # BOM
    "⁠"      # word joiner
)
_L1_TO_SPACE = "‌"     # ZWNJ is a word boundary, not nothing

# L2: editorial sigla and press conventions.
_L2_DELETE_RANGES = [
    [0x0610, 0x061B],   # Arabic signs through U+061A; upper bound is exclusive
]
_L2_DELETE = "ٴ"       # high hamza, used as an izafat marker
_L2_MAP = {
    # hamza-e-izafat carriers: دیدۂ / دیدہ, حلقۂ / حلقہ, گریۂ / گریہ
    "ۂ": "ہ", "ۃ": "ہ",
    # hamza carriers: مئے / مے, سوئے / سوے, ہوؤں / ہووں
    "ئ": "ی", "ؤ": "و",
    # bari ye WITH hamza -> bari ye. Note: stays bari ye, is NOT collapsed to
    # choti ye, because ے / ی is a real distinction (ہے vs ہی).
    "ۓ": "ے",
}
# Mapping ئ -> ی leaves a doubled ye (مئے -> میے vs مے). Collapse runs.
_L2_REGEX = [[r"یی+", "ی"], [r"ےے+", "ے"], [r"یے", "ے"]]
_L2_TO_SPACE = (
    "«»؛،؟!?.,:;()[]{}\"'—–-*/\\|_=+<>#%&@~`^$٪۔٬…"
    "“”„‟‘’‹›"
)

# L3: vowel marks. Editorially significant (see module docstring), own rung.
_L3_DELETE_RANGES = [
    [0x064B, 0x0653],   # fathatan..shadda, sukun; upper bound is exclusive
    [0x0654, 0x065A],   # hamza above/below, superscript marks through U+0659
]
_L3_DELETE = "ٰ"       # superscript alef

# L5: phonemic folds. RETRIEVAL ONLY. Every one of these can change a word.
_L5_MAP = {
    "ٹ": "ت", "ڈ": "د", "ڑ": "ر",     # retroflex: ٹال/تال, ڈال/دال, گڑ/گر
    "ں": "ن",                           # noon ghunna: میں/مین
    "ے": "ی",                           # bari ye: ہے/ہی
    "ھ": "ہ",                           # do-chashmi he: کھا/کہا, بھار/بہار
    "آ": "ا",                           # alef madda: آب/اب
    "ژ": "ز",
}
_L5_REGEX = [[r"یی+", "ی"]]
_L5_DIGITS = r"[۰-۹٠-٩0-9]+"

# ---------------------------------------------------------------- apparatus-v2 additions
# THE SAME NUMERAL IN TWO UNICODE BLOCKS. U+06F0..06F9 (EXTENDED ARABIC-INDIC, the Urdu
# forms ۰۱۲۳۴۵۶۷۸۹) and U+0660..0669 (ARABIC-INDIC ٠١٢٣٤٥٦٧٨٩) encode the same ten digits, and
# ASCII 0-9 encodes them again. Which block a reader emits is a font and keyboard artifact
# carrying no textual information whatsoever -- ۱۴۸ and ١٤٨ are both the number 148.
#
# apparatus-v1 does not fold them, and on ARSHI-1958 that turned out to account for 16.2% of
# all lines by itself, with 19.5% resolved once combined with the other folds. Printed numbers
# are everywhere in this corpus -- verse numbers, page references, dates, apparatus locators --
# so leaving them unfolded reports a spurious variant at every one of them.
#
# This belongs at the ENCODING rung, which exists precisely for differences that are not about
# the text. It is added as apparatus-v2 rather than folded into apparatus-v1.
_DIGIT_MAP = {}
for _i in range(10):
    _DIGIT_MAP[chr(0x06F0 + _i)] = chr(0x0660 + _i)   # Urdu/Persian -> Arabic-Indic
    _DIGIT_MAP[chr(0x30 + _i)] = chr(0x0660 + _i)     # ASCII -> Arabic-Indic

APPARATUS_SPEC = {
    "name": "apparatus", "version": 1,
    "doc": "conservative; reports shallowest agreement level; erases nothing",
    "markup": _MARKUP,
    "levels": [
        {"name": "encoding", "map": _L1_MAP, "delete": _L1_DELETE,
         "to_space": _L1_TO_SPACE, "nfkc": True},
        {"name": "orthographic", "map": _L2_MAP, "delete": _L2_DELETE,
         "delete_ranges": _L2_DELETE_RANGES, "to_space": _L2_TO_SPACE,
         "regex": _L2_REGEX},
        {"name": "vocalisation", "delete": _L3_DELETE,
         "delete_ranges": _L3_DELETE_RANGES},
        {"name": "segmentation", "drop_spaces": True},
    ],
}

RETRIEVAL_SPEC = {
    "name": "retrieval", "version": 1,
    "doc": "aggressive; candidate discovery only; deliberately folds phonemic "
           "distinctions. Frozen to reproduce the behaviour the 2026-09-04 gold "
           "sets were built with.",
    "markup": _MARKUP,
    "levels": (APPARATUS_SPEC["levels"][:3]
               + [{"name": "phonemic", "map": _L5_MAP, "regex": _L5_REGEX,
                   "digits": _L5_DIGITS, "digits_to": " 0 "},
                  {"name": "segmentation", "drop_spaces": True}]),
}


def _hash(spec):
    """Deterministic hash of the RULES. Comments and refactors do not move it."""
    blob = json.dumps(spec, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


APPARATUS_HASH = _hash(APPARATUS_SPEC)
RETRIEVAL_HASH = _hash(RETRIEVAL_SPEC)
APPARATUS_ID = f"apparatus-v{APPARATUS_SPEC['version']}+{APPARATUS_HASH}"

RETRIEVAL_ID = f"retrieval-v{RETRIEVAL_SPEC['version']}+{RETRIEVAL_HASH}"

# ---------------------------------------------------------------- executor


def _strip_markup(t, markup):
    t = re.sub(markup["leading_tag"], "", t or "")
    t = re.sub(markup["placeholder"], " ", t, flags=re.I)
    return re.sub(markup["uncertain"], r"\1", t)


def _apply(t, level):
    if level.get("nfkc"):
        t = unicodedata.normalize("NFKC", t)
    for ch in level.get("delete", ""):
        t = t.replace(ch, "")
    for lo, hi in level.get("delete_ranges", []):
        t = t.translate(dict.fromkeys(range(lo, hi), None))
    for ch in level.get("to_space", ""):
        t = t.replace(ch, " ")
    m = level.get("map")
    if m:
        t = t.translate(str.maketrans(m))
    for pat, rep in level.get("regex", []):
        t = re.sub(pat, rep, t)
    if level.get("digits"):
        t = re.sub(level["digits"], level.get("digits_to", " 0 "), t)
    if level.get("drop_spaces"):
        return "".join(t.split())
    return " ".join(t.split())


def normalize(text, spec, upto=None):
    """Apply `spec` cumulatively up to and including level `upto` (name or index)."""
    t = _strip_markup(text, spec["markup"])
    names = [l["name"] for l in spec["levels"]]
    last = len(names) - 1 if upto is None else (
        names.index(upto) if isinstance(upto, str) else upto)
    for level in spec["levels"][:last + 1]:
        t = _apply(t, level)
    return t


APPARATUS_V2_SPEC = {
    "name": "apparatus", "version": 2,
    "doc": "as v1, plus digit-block folding at the encoding rung; erases nothing textual",
    "markup": _MARKUP,
    "levels": [
        {"name": "encoding", "map": {**_L1_MAP, **_DIGIT_MAP}, "delete": _L1_DELETE,
         "to_space": _L1_TO_SPACE, "nfkc": True},
        {"name": "orthographic", "map": _L2_MAP, "delete": _L2_DELETE,
         "delete_ranges": _L2_DELETE_RANGES, "to_space": _L2_TO_SPACE,
         "regex": _L2_REGEX},
        {"name": "vocalisation", "delete": _L3_DELETE,
         "delete_ranges": _L3_DELETE_RANGES},
        {"name": "segmentation", "drop_spaces": True},
    ],
}

APPARATUS_V2_HASH = _hash(APPARATUS_V2_SPEC)
APPARATUS_V2_ID = f"apparatus-v{APPARATUS_V2_SPEC['version']}+{APPARATUS_V2_HASH}"

LEVELS = [l["name"] for l in APPARATUS_SPEC["levels"]]


def _agree(a, b, spec):
    if a == b:
        return "identical"
    for level in spec["levels"]:
        name = level["name"]
        if normalize(a, spec, name) == normalize(b, spec, name):
            return name
    return None


def agreement_level(a, b):
    """Shallowest apparatus-v1 level at which two readings agree, else None.

    None is the decision-relevant answer: the readings differ in substance, so the
    difference is either a reader error or a genuine textual variant and needs
    evidence -- a footnote, another witness, or scansion -- to resolve.

    Kept byte-for-byte in behaviour: every comparison already recorded cites v1.
    """
    return _agree(a, b, APPARATUS_SPEC)


def agreement_level_v2(a, b):
    """As above under apparatus-v2, which additionally folds digit blocks at `encoding`."""
    return _agree(a, b, APPARATUS_V2_SPEC)


def apparatus(text, level="segmentation"):
    return normalize(text, APPARATUS_SPEC, level)


def retrieval_tokens(text):
    """Token list for retrieval. Spaces are boundaries, not dropped."""
    return normalize(text, RETRIEVAL_SPEC, "phonemic").split()


def retrieval_ident(text):
    """Spaceless retrieval form: word identity ignoring segmentation."""
    return normalize(text, RETRIEVAL_SPEC, "segmentation")


def retrieval_merges(a, b):
    return retrieval_tokens(a) == retrieval_tokens(b)


if __name__ == "__main__":
    print(RETRIEVAL_ID)
    print(APPARATUS_ID)
    print(f"apparatus levels: {' -> '.join(LEVELS)}")
