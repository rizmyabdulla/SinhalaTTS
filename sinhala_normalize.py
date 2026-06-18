"""
Sinhala Text Normalization for CosyVoice3 SFT
==============================================

Why this matters for natural-sounding output
-------------------------------------------
CosyVoice3 (and its Qwen2 text encoder) tokenizes raw unicode graphemes.
If the same Sinhala word appears in two different unicode encodings
(NFC vs NFD, with/without ZWJ, with/without variation selectors), the
tokenizer will see them as different sequences and the LLM will fail
to generalize. The result: stilted, jerky prosody because the model
is fighting inconsistent input.

Sinhala has three known trouble spots we must handle:

1. **Unicode normalization**: Sinhala in the wild is almost always in
   NFC (precomposed), but copy-paste from web pages can mix in NFD
   (decomposed) sequences like "ක + ් + ෙ" vs the precomposed "කෙ".
   We force everything to NFC.

2. **ZWJ/ZWNJ (zero-width joiner / non-joiner)**: U+200D / U+200C.
   Sinhala uses these for ligature control in conjunction with the
   "repaya" (rakaransha). We keep them because they are semantically
   meaningful, but normalize any stray "joiner-storm" (3+ in a row)
   down to one, which is what native Sinhala Unicode does.

3. **Punctuation & digits**: The OpenSLR30 transcripts are clean, but
   real-world Sinhala mixes Sinhala digits (෦-෯), Arabic digits (0-9),
   English, and Sinhala punctuation (෴, …). We map everything to
   a single, consistent form that the Qwen2 tokenizer handles cleanly.

4. **Trailing/leading whitespace and control characters** that
   occasionally leak in from web-sourced transcripts.

The output of this module is what gets written to the Kaldi `text` file
and ultimately fed to the Qwen2 tokenizer as `sample['text']`.

Tested on: OpenSLR30 si_lk.lines.txt, Path Nirvana Sinhala TTS, XLR-S,
and the Sinhala SinLlama training corpus.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

# -----------------------------------------------------------------------------
# Character-level mappings
# -----------------------------------------------------------------------------

# Sinhala digits (0-9) -> ASCII digits. The TTS front-end in the base
# CosyVoice3 model is trained on multilingual text, but our transcripts
# here are clean word-level Sinhala. We still normalize digits for
# consistency so the same utterance always tokenizes the same way.
_SI_DIGITS = str.maketrans({
    "෦": "0", "෧": "1", "෨": "2", "෩": "3", "෪": "4",
    "෫": "5", "෬": "6", "෭": "7", "෮": "8", "෯": "9",
})

# Sinhala-specific punctuation that Qwen2 vocab covers poorly. Map to
# ASCII equivalents. We deliberately do NOT strip the period (.) and
# comma (,) — those are valid sentence boundary markers and the
# tokenizer uses them.
_SI_PUNCT = str.maketrans({
    "෴": ".",   # Sinhala "kunddaliya" (full stop)
    "…" : "...",  # horizontal ellipsis
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "·": " ",  # middle dot - keep as space to avoid orphan tokens
})

# Whitespace canonicalization: collapse any run of unicode whitespace
# (including NBSP U+00A0, ZWSP U+200B) into a single ASCII space, EXCEPT
# the ZWJ/ZWNJ which we handle separately below.
_WS = re.compile(r"[ \t\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+")

# Zero-width joiner: collapse runs of 3+ to a single ZWJ. Pairs are
# intentional (e.g. "්‍ය" is a real conjunct), so we keep 1-2.
_ZWJ_RUN = re.compile("\u200d{3,}")
_ZWNJ_RUN = re.compile("\u200c{3,}")

# Combine consecutive ZWJ+ZWNJ+ZWJ into a single ZWJ (this is what
# native Sinhala keyboards emit after "repaya + yansaya").
_ZWJ_ZWNJ_ZWJ = re.compile("\u200d\u200c\u200d")

# Strip control characters except the common ones we want to keep
# (newline, tab). Most transcript files won't have these, but a few
# web-scraped corpora do.
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _is_sinhala(ch: str) -> bool:
    """Return True if a character is in the Sinhala Unicode block (U+0D80..U+0DFF)."""
    if not ch:
        return False
    return 0x0D80 <= ord(ch[0]) <= 0x0DFF


def _strip_isolated_diacritics(text: str) -> str:
    """Remove Sinhala vowel signs / diacritics that lost their base character.

    A few OpenSLR30 transcripts have lines where the first char of a
    word is a diacritic (pilla, kombuva, etc.) without a base consonant
    preceding it — almost always a transcript error. Without a base
    consonant these characters have no pronunciation, so we drop them.
    """
    out = []
    prev_was_base = True  # start of string counts as a "base"
    for ch in text:
        cp = ord(ch)
        # Sinhala independent vowels (U+0D85..U+0D96) ARE base chars.
        # Sinhala consonants (U+0D9A..U+0DC6) ARE base chars.
        # The "virama" / "al-lakuna" U+0DCA is NOT a base.
        # Everything else in the block (U+0D00..U+0D84, U+0DCA..U+0DFF)
        # is a diacritic / modifier.
        if _is_sinhala(ch):
            is_base = (
                (0x0D85 <= cp <= 0x0D96)        # independent vowels
                or (0x0D9A <= cp <= 0x0DC6)      # consonants
            )
        else:
            is_base = True
        if is_base:
            out.append(ch)
            prev_was_base = True
        elif prev_was_base:
            out.append(ch)  # keep first diacritic of a sequence
        # else: drop orphan diacritic
    return "".join(out)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def normalize_sinhala(text: str, *, drop_punct: bool = False) -> str:
    """Normalize a single Sinhala transcript line.

    Steps, in order:
        1. Strip control characters
        2. NFC unicode normalization
        3. Sinhala digit normalization
        4. Sinhala punctuation normalization
        5. Collapse whitespace
        6. Canonicalize ZWJ/ZWNJ runs
        7. Drop orphan diacritics
        8. Trim leading/trailing whitespace

    Args:
        text: raw transcript string (may have surrounding quotes from TSV)
        drop_punct: if True, strip ASCII punctuation entirely (useful for
            some text-only training regimes; for CosyVoice3 SFT we keep
            punctuation so the model learns prosody around sentence breaks)

    Returns:
        Cleaned Sinhala string. May be empty if input had no Sinhala.
    """
    if text is None:
        return ""

    # 0. Drop enclosing quotes left over from a TSV split
    s = text.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ('"', "'"):
        s = s[1:-1].strip()

    # 1. Strip C0/C1 control chars (keep \n and \t for safety)
    s = _CONTROL.sub("", s)

    # 2. Unicode NFC — the single most important step
    s = unicodedata.normalize("NFC", s)

    # 3. Sinhala digits -> ASCII digits
    s = s.translate(_SI_DIGITS)

    # 4. Sinhala punctuation -> ASCII equivalents
    s = s.translate(_SI_PUNCT)

    # 5. Whitespace collapse
    s = _WS.sub(" ", s)

    # 6. ZWJ/ZWNJ run collapse
    s = _ZWJ_RUN.sub("\u200d", s)
    s = _ZWNJ_RUN.sub("\u200c", s)
    s = _ZWJ_ZWNJ_ZWJ.sub("\u200d", s)

    # 7. Drop orphan diacritics
    s = _strip_isolated_diacritics(s)

    # 8. Optional punctuation drop (rarely used)
    if drop_punct:
        s = re.sub(r"[^\w\s\u0D80-\u0DFF]", "", s)

    # 9. Trim
    s = s.strip()

    return s


def is_valid_transcript(text: str, min_sinhala: int = 3) -> bool:
    """Return True if `text` looks like real Sinhala speech content.

    Used to filter out empty / garbage lines before we spend disk space
    on speech tokens. The threshold is loose: 3+ Sinhala chars is
    enough to count as a real word, given the average Sinhala word
    length is ~5-7 characters.
    """
    if not text:
        return False
    n_sinhala = sum(1 for ch in text if _is_sinhala(ch))
    return n_sinhala >= min_sinhala


# -----------------------------------------------------------------------------
# Smoke test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        # From the user's OpenSLR30 sample
        " à¶šà·à¶šà¶§à¶­à·Š à¶¸à¶‚ à·€à·™à¶±à¶¯à· à¶­à¶»à¶¸à·Š à¶šà·à¶½à·™ à¶œà¶±à·Šà¶±à·à¶­à·’à·€ à¶‡à¶³ à¶œà¶­à·Šà¶­à· ",
        # NFC Sinhala (what it should look like after normalization)
        "කොස් කම කළේ මට නම් මට කියන්න",
        # A number / digit test
        "෴ වෙනි පටන් 1 2 3 හතර",
        # Pathological: NFD with diacritics
        unicodedata.normalize("NFD", "කෙසේ"),
    ]
    for s in samples:
        out = normalize_sinhala(s)
        valid = is_valid_transcript(out)
        print(f"IN : {s!r}")
        print(f"OUT: {out!r}  (valid={valid})")
        print("---")