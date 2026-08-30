"""Text and stylometric features.

Three families live here:

1. LinguisticFeatures -- ~30 hand-crafted stylometric signals. These encode
   what actually separates paid text from lived experience: SPECIFICITY.
   A real customer writes "measured 31cm not the 29cm listed"; a paid one
   writes "amazing quality, highly recommend". So numerals, measurement units,
   comparatives and hedges are positive evidence of authenticity, while
   superlative and intensifier density is evidence against it.

2. TF-IDF word n-grams -- topical/phrasal giveaways and template fragments.

3. TF-IDF character n-grams -- robust to the paraphrasing and typo-injection
   that review farms use to dodge exact-duplicate filters, because character
   n-grams survive small edits that destroy word n-grams.

All transformers are plain classes (no lambdas, no closures) so the fitted
pipeline pickles cleanly for serving.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from ..config import TextFeatureConfig
from ..schema import HELPFUL, RATING, TEXT, TIMESTAMP, TITLE, VERIFIED

# --- lexicons --------------------------------------------------------------

SUPERLATIVES = {
    "best", "greatest", "perfect", "amazing", "incredible", "outstanding",
    "excellent", "superb", "phenomenal", "flawless", "unbeatable", "ultimate",
    "awesome", "fantastic", "wonderful", "brilliant", "exceptional",
}
INTENSIFIERS = {
    "very", "really", "so", "absolutely", "totally", "completely", "extremely",
    "super", "highly", "definitely", "truly", "utterly", "incredibly",
}
HEDGES = {
    "though", "however", "although", "but", "albeit", "maybe", "perhaps",
    "somewhat", "slightly", "arguably", "seems", "seemed", "probably",
    "honestly", "admittedly",
}
COMPARATIVES = {
    "than", "compared", "versus", "vs", "instead", "unlike", "whereas",
    "better", "worse", "cheaper", "lighter", "heavier", "sturdier",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
SECOND_PERSON = {"you", "your", "yours", "yourself"}
PROMO_PHRASES = [
    "highly recommend", "must buy", "worth every penny", "five stars",
    "do not hesitate", "best on the market", "will buy again",
    "you need this", "look no further", "game changer", "10/10",
]

_UNIT_RE = re.compile(
    r"\b\d+(\.\d+)?\s?(cm|mm|m|inch|in|ft|kg|g|lb|oz|ml|l|hours?|hrs?|"
    r"mins?|minutes?|days?|weeks?|months?|years?|w|v|mah|gb|tb|mhz|ghz)\b",
    re.I,
)
_NUM_RE = re.compile(r"\d")
_WORD_RE = re.compile(r"[a-z']+")
_SENT_RE = re.compile(r"[.!?]+")
_REPEAT_RE = re.compile(r"(.)\1{2,}")          # "sooooo"
_NONASCII_RE = re.compile(r"[^\x00-\x7F]")     # emoji / decorative glyphs

# Base names used internally; the exported columns carry an `lx_` prefix so
# the linguistic block can never collide with a schema column (`rating` and
# `helpful_votes` appear in both) when both are joined onto one frame.
_LING_BASE: list[str] = [
    "char_len", "word_count", "sent_count", "avg_word_len", "avg_sent_len",
    "type_token_ratio", "exclam_count", "exclam_per_sent", "question_count",
    "upper_char_ratio", "allcaps_words", "digit_char_ratio", "punct_ratio",
    "numeral_count", "unit_mentions", "specificity_score",
    "superlative_count", "superlative_density", "intensifier_count",
    "intensifier_density", "hedge_count", "hedge_density",
    "comparative_count", "first_person_ratio", "second_person_ratio",
    "promo_phrase_count", "repeat_char_runs", "nonascii_count",
    "title_len", "title_word_count", "title_text_len_ratio",
    "long_word_share", "unique_word_count",
    # metadata (non-text but naturally numeric and always present)
    "rating", "rating_is_extreme", "rating_is_five", "rating_is_one",
    "verified", "helpful_votes", "log_helpful",
    "hour", "is_night", "dayofweek", "is_weekend",
    # cross-signal: gushing text on a low rating, or terse text on a 5
    "sentiment_rating_mismatch", "text_len_per_star",
]

LINGUISTIC_COLUMNS: list[str] = [f"lx_{c}" for c in _LING_BASE]


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """DataFrame -> dense float32 matrix of stylometric + metadata features."""

    def fit(self, X: pd.DataFrame, y=None) -> "LinguisticFeatures":
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.asarray(LINGUISTIC_COLUMNS, dtype=object)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        texts = X[TEXT].fillna("").astype(str).tolist()
        titles = (
            X[TITLE].fillna("").astype(str).tolist()
            if TITLE in X.columns else [""] * len(texts)
        )
        ratings = X[RATING].to_numpy(dtype=np.float64)
        verified = (
            X[VERIFIED].fillna(False).astype(bool).to_numpy()
            if VERIFIED in X.columns else np.zeros(len(texts), bool)
        )
        helpful = (
            X[HELPFUL].fillna(0).to_numpy(dtype=np.float64)
            if HELPFUL in X.columns else np.zeros(len(texts))
        )
        ts = pd.to_datetime(X[TIMESTAMP], utc=True)
        hours = ts.dt.hour.to_numpy()
        dows = ts.dt.dayofweek.to_numpy()

        out = np.zeros((len(texts), len(_LING_BASE)), dtype=np.float32)
        idx = {c: i for i, c in enumerate(_LING_BASE)}

        for i, raw in enumerate(texts):
            low = raw.lower()
            words = _WORD_RE.findall(low)
            wc = len(words)
            char_len = len(raw)
            sents = [s for s in _SENT_RE.split(raw) if s.strip()]
            sc = max(len(sents), 1)
            uniq = len(set(words))

            sup = sum(1 for w in words if w in SUPERLATIVES)
            inten = sum(1 for w in words if w in INTENSIFIERS)
            hedge = sum(1 for w in words if w in HEDGES)
            comp = sum(1 for w in words if w in COMPARATIVES)
            fp = sum(1 for w in words if w in FIRST_PERSON)
            spp = sum(1 for w in words if w in SECOND_PERSON)
            numerals = len(_NUM_RE.findall(raw))
            units = len(_UNIT_RE.findall(raw))
            promo = sum(1 for ph in PROMO_PHRASES if ph in low)
            exclam = raw.count("!")

            wc_safe = max(wc, 1)
            out[i, idx["char_len"]] = char_len
            out[i, idx["word_count"]] = wc
            out[i, idx["sent_count"]] = len(sents)
            out[i, idx["avg_word_len"]] = (
                np.mean([len(w) for w in words]) if words else 0.0
            )
            out[i, idx["avg_sent_len"]] = wc / sc
            out[i, idx["type_token_ratio"]] = uniq / wc_safe
            out[i, idx["exclam_count"]] = exclam
            out[i, idx["exclam_per_sent"]] = exclam / sc
            out[i, idx["question_count"]] = raw.count("?")
            out[i, idx["upper_char_ratio"]] = (
                sum(1 for ch in raw if ch.isupper()) / max(char_len, 1)
            )
            out[i, idx["allcaps_words"]] = sum(
                1 for w in raw.split() if len(w) > 2 and w.isupper()
            )
            out[i, idx["digit_char_ratio"]] = numerals / max(char_len, 1)
            out[i, idx["punct_ratio"]] = (
                sum(1 for ch in raw if ch in ".,;:!?-()") / max(char_len, 1)
            )
            out[i, idx["numeral_count"]] = numerals
            out[i, idx["unit_mentions"]] = units
            # Specificity: concrete detail per unit of length. The single most
            # useful hand-crafted signal in practice.
            out[i, idx["specificity_score"]] = (
                (numerals + 2 * units + comp) / np.log1p(wc_safe)
            )
            out[i, idx["superlative_count"]] = sup
            out[i, idx["superlative_density"]] = sup / wc_safe
            out[i, idx["intensifier_count"]] = inten
            out[i, idx["intensifier_density"]] = inten / wc_safe
            out[i, idx["hedge_count"]] = hedge
            out[i, idx["hedge_density"]] = hedge / wc_safe
            out[i, idx["comparative_count"]] = comp
            out[i, idx["first_person_ratio"]] = fp / wc_safe
            out[i, idx["second_person_ratio"]] = spp / wc_safe
            out[i, idx["promo_phrase_count"]] = promo
            out[i, idx["repeat_char_runs"]] = len(_REPEAT_RE.findall(raw))
            out[i, idx["nonascii_count"]] = len(_NONASCII_RE.findall(raw))
            out[i, idx["long_word_share"]] = (
                sum(1 for w in words if len(w) >= 8) / wc_safe
            )
            out[i, idx["unique_word_count"]] = uniq

            title = titles[i]
            t_words = _WORD_RE.findall(title.lower())
            out[i, idx["title_len"]] = len(title)
            out[i, idx["title_word_count"]] = len(t_words)
            out[i, idx["title_text_len_ratio"]] = len(title) / max(char_len, 1)

            r = ratings[i]
            out[i, idx["rating"]] = r
            out[i, idx["rating_is_extreme"]] = float(r <= 1.5 or r >= 4.5)
            out[i, idx["rating_is_five"]] = float(r >= 4.5)
            out[i, idx["rating_is_one"]] = float(r <= 1.5)
            out[i, idx["verified"]] = float(verified[i])
            out[i, idx["helpful_votes"]] = helpful[i]
            out[i, idx["log_helpful"]] = np.log1p(max(helpful[i], 0))
            out[i, idx["hour"]] = hours[i]
            out[i, idx["is_night"]] = float(hours[i] < 6 or hours[i] >= 23)
            out[i, idx["dayofweek"]] = dows[i]
            out[i, idx["is_weekend"]] = float(dows[i] >= 5)

            # Positive-word load that disagrees with the star rating: a common
            # signature of hastily written incentivised text.
            pos_load = (sup + inten) / wc_safe
            out[i, idx["sentiment_rating_mismatch"]] = (
                pos_load * (1.0 if r <= 2.5 else 0.0)
                + (0.0 if r <= 2.5 else max(0.0, 0.15 - pos_load))
            )
            out[i, idx["text_len_per_star"]] = char_len / max(r, 1.0)

        return out


class TextColumnSelector(BaseEstimator, TransformerMixin):
    """Pull one text column out as a 1-D list for a vectoriser.

    A named class rather than a FunctionTransformer with a lambda, so the
    fitted pipeline is picklable.
    """

    def __init__(self, column: str = TEXT, join_title: bool = True):
        self.column = column
        self.join_title = join_title

    def fit(self, X: pd.DataFrame, y=None) -> "TextColumnSelector":
        return self

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        # The vectoriser downstream names the real features; this exists only
        # so `Pipeline.get_feature_names_out()` can traverse the chain, which
        # the token-level explainer depends on.
        return np.asarray([self.column], dtype=object)

    def transform(self, X: pd.DataFrame) -> list[str]:
        body = X[self.column].fillna("").astype(str)
        if self.join_title and TITLE in X.columns:
            # Titles carry signal and are short; prepending is cheaper than a
            # fourth vectoriser.
            return (X[TITLE].fillna("").astype(str) + " . " + body).tolist()
        return body.tolist()


def make_word_vectorizer(cfg: TextFeatureConfig | None = None) -> TfidfVectorizer:
    cfg = cfg or TextFeatureConfig()
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=cfg.word_ngram,
        max_features=cfg.word_max_features,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        strip_accents="unicode",
        lowercase=True,
    )


def make_char_vectorizer(cfg: TextFeatureConfig | None = None) -> TfidfVectorizer:
    cfg = cfg or TextFeatureConfig()
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=cfg.char_ngram,
        max_features=cfg.char_max_features,
        min_df=cfg.min_df,
        sublinear_tf=cfg.sublinear_tf,
        lowercase=True,
    )
