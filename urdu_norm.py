#!/usr/bin/env python3
"""Urdu Perso-Arabic normalization, in two strengths.

siraj/ used ONE normalization for both anchoring and CER scoring. Urdu needs two,
because the distinctions that are most OCR-fragile in lithographed nasta'liq are
phonemic in Urdu and not in Persian:

  retroflex ٹ ڈ ڑ vs plain ت د ر   -- four dots vs none; routinely lost in litho
  noon ghunna ں vs ن                -- nasal marking; routinely lost
  bari ye ے vs choti ye ی           -- positional variants a reader may swap
  do-chashmi he ھ vs he ہ           -- aspiration; کھ vs ک is a different word

MATCH strength folds all of these together: it is used only to FIND a candidate
pair, where a false negative costs a lost gold pair and a fold-in costs almost
nothing (the model judge re-decides anyway).

SCORE strength keeps them: it is used to compute CER on a confirmed pair, where
folding them in would silently forgive real reading errors and flatter the reader.

Using MATCH for scoring would understate CER. Using SCORE for matching loses
true pairs. Do not collapse these back into one function.
"""
import re
import unicodedata

# Combining marks: harakat, shadda, sukun, superscript alef, hamza above/below, and
# U+0610-U+061A -- which includes **U+0614 ARABIC SIGN TAKHALLUS**, the mark printed over
# a pen-name (غالبؔ, اسدؔ, امیدؔ). Missing it made every takhallus look like a different
# word from the reference's, and it accounted for 6 of 35 apparent "lexical" divergences
# on the first Hamid Ali Khan run.
_MARKS = dict.fromkeys(
    list(range(0x064B, 0x0653)) + list(range(0x0610, 0x061B))
    + [0x0654, 0x0655, 0x0670, 0x0656, 0x0657, 0x0658, 0x0674],
    None,
)

_PUNCT = dict.fromkeys(
    map(ord, "«»؛،؟!?.,:;()[]{}\"'—–-*/\\|_=+<>#%&@~`^$٪۔٬…"
        "\u201c\u201d\u201e\u201f\u2018\u2019\u2039\u203a"), " "
)

# invisibles and joiners: tatweel, ZWNJ, ZWJ, RLM, LRM, ZWSP, soft hyphen, BOM
_INVISIBLE = str.maketrans({
    "ـ": "", "‌": " ", "‍": "", "‎": "", "‏": "",
    "​": "", "­": "", "﻿": "", "⁠": "",
})

# Shared by both strengths: variants that are pure encoding noise, never a
# meaning difference in Urdu orthography.
_BASE = str.maketrans({
    "ي": "ی", "ى": "ی", "ﻯ": "ی",     # Arabic/final ye -> Urdu choti ye
    "ك": "ک", "ﻙ": "ک",                # Arabic kaf -> Urdu kaf
    "ه": "ہ",                           # Arabic he -> Urdu he goal
    "ة": "ہ", "ۀ": "ہ",
    # hamza-e-izafat carriers: دیدۂ / دیدہ, حلقۂ / حلقہ, گریۂ / گریہ. The hamza marks
    # the izafat, which is a typesetting choice, not a different word.
    "ۂ": "ہ", "ۃ": "ہ",
    "ﻻ": "لا", "ﷲ": "اللہ",
    "أ": "ا", "إ": "ا", "ٱ": "ا",
    "ؤ": "و",                           # hamza-carrier waw
    "ئ": "ی",                           # hamza-carrier ye
    "ﮪ": "ہ", "ﮭ": "ھ",
})

# MATCH-only: OCR-fragile but phonemic. Folded in for retrieval, kept for CER.
_FOLD = str.maketrans({
    "ٹ": "ت", "ڈ": "د", "ڑ": "ر",     # retroflex -> plain (four dots vs none)
    "ں": "ن",                           # noon ghunna -> noon
    "ے": "ی", "ۓ": "ی",                 # bari ye (+hamza) -> choti ye
    "ھ": "ہ",                           # do-chashmi he -> he
    "آ": "ا",                           # alef madda -> alef
    "ژ": "ز", "ۃ": "ہ",
})

_TAG = re.compile(r"^\s*\[[^\]]{0,40}\]\s*")            # leading [Urdu] / [Mixed: ...]
_MARK = re.compile(r"\[(?:ILL|\?|BLANK PAGE|PLATE|ORNAMENT)\]", re.I)
_GUESS = re.compile(r"\[([^\]]*?)\??\]")                 # [word?] -> word, keep the guess
_DIGITS = re.compile(r"[۰-۹٠-٩0-9]+")


def _clean(t: str) -> str:
    t = _MARK.sub(" ", _TAG.sub("", t or ""))
    t = _GUESS.sub(r"\1", t)
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(_INVISIBLE).translate(_MARKS).translate(_PUNCT)
    return t.translate(_BASE)


# Mapping the hamza carrier ئ -> ی leaves a doubled ye: مئے becomes میی where مے is
# می, and سوئے becomes سویی where سوے is سوی. Both spellings are the same word in
# different press conventions, so collapse runs of ye at MATCH strength only.
_YY = re.compile(r"یی+")


def norm_match(t: str) -> list[str]:
    """Aggressive: for anchoring and retrieval only. Returns a token list."""
    t = _clean(t).translate(_FOLD)
    t = _YY.sub("ی", t)
    t = _DIGITS.sub(" 0 ", t)
    return t.split()


def norm_ident(t: str) -> str:
    """Spaceless MATCH form: word identity ignoring segmentation.

    Word division is not orthography in Perso-Arabic print -- درخور / در خور,
    صدہزار / صد ہزار, بیوفا / بے وفا, سخنور / سخن ور are the same words differently
    spaced. For a critical apparatus that difference must NOT count as a textual
    variant, or the variant list fills with typesetting and the real readings are lost
    in it. Use this to decide whether two hemistichs say the same thing; use
    norm_score to record exactly how each witness spells it.
    """
    return "".join(norm_match(t))


def norm_score(t: str) -> list[str]:
    """Conservative: for CER on a confirmed pair. Keeps phonemic distinctions."""
    t = _DIGITS.sub(" 0 ", _clean(t))
    return t.split()


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


try:
    from rapidfuzz.distance import Levenshtein as _RF
    levenshtein = _RF.distance          # same metric, C speed
except ImportError:
    pass


def cer(hyp: str, ref: str) -> float:
    """Character error rate, normalized by the longer string (siraj convention)."""
    if not hyp and not ref:
        return 0.0
    return levenshtein(hyp, ref) / max(len(hyp), len(ref))
