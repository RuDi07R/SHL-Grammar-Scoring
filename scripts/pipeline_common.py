from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
ORIGINAL_COLUMNS = [
    "word_count", "unique_word_count", "sentence_count", "avg_sentence_length",
    "sentence_length_std", "type_token_ratio", "char_count", "avg_word_length",
    "repetition_ratio", "filler_count", "question_count", "exclamation_count",
]
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
FILLERS = {"um", "uh", "erm", "hmm", "like", "basically", "actually"}
CLAUSE_DEPS = {"advcl", "ccomp", "xcomp", "relcl", "conj"}
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART"}
CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}


def load_nlp():
    return spacy.load("en_core_web_sm", disable=["ner"])


def _safe_ratio(value: float, denominator: int) -> float:
    return float(value / denominator) if denominator else 0.0


def extract_v2_row(text: str, nlp) -> dict[str, float]:
    text = "" if pd.isna(text) else str(text)
    doc = nlp(text)
    tokens = [token for token in doc if not token.is_space and not token.is_punct]
    words = [token.text.lower() for token in tokens if token.is_alpha]
    word_count = len(words)
    counts = {pos: sum(token.pos_ == pos for token in tokens) for pos in [
        "NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP", "DET", "CCONJ", "SCONJ", "AUX", "MODAL"
    ]}
    counts["MODAL"] = sum(token.tag_ in {"MD"} for token in tokens)
    sentences = list(doc.sents)
    sentence_count = max(len(sentences), 1)
    unique_count = len(set(words))
    frequencies = pd.Series(words).value_counts() if words else pd.Series(dtype=int)
    hapax_count = int((frequencies == 1).sum())
    long_count = sum(len(word) >= 8 for word in words)
    adjacent_repeats = sum(left == right for left, right in zip(words, words[1:]))
    bigrams = list(zip(words, words[1:]))
    trigrams = list(zip(words, words[1:], words[2:]))
    repeated_phrases = sum(left == right for left, right in zip(bigrams, bigrams[1:]))
    repeated_phrases += sum(left == right for left, right in zip(trigrams, trigrams[1:]))
    filler_count = sum(word in FILLERS for word in words)
    clause_like = sum(token.dep_ in CLAUSE_DEPS for token in tokens)
    conjunction_count = counts["CCONJ"] + counts["SCONJ"]
    dependency_depth = []
    for token in tokens:
        depth = 0
        head = token
        while head.head is not head and depth < 50:
            depth += 1
            head = head.head
        dependency_depth.append(depth)

    features = {
        "noun_count": counts["NOUN"], "verb_count": counts["VERB"],
        "adjective_count": counts["ADJ"], "adverb_count": counts["ADV"],
        "pronoun_count": counts["PRON"], "preposition_count": counts["ADP"],
        "determiner_count": counts["DET"], "conjunction_count": conjunction_count,
        "auxiliary_count": counts["AUX"], "modal_count": counts["MODAL"],
        "noun_ratio": _safe_ratio(counts["NOUN"], word_count),
        "verb_ratio": _safe_ratio(counts["VERB"], word_count),
        "adjective_ratio": _safe_ratio(counts["ADJ"], word_count),
        "adverb_ratio": _safe_ratio(counts["ADV"], word_count),
        "pronoun_ratio": _safe_ratio(counts["PRON"], word_count),
        "preposition_ratio": _safe_ratio(counts["ADP"], word_count),
        "determiner_ratio": _safe_ratio(counts["DET"], word_count),
        "conjunction_ratio": _safe_ratio(conjunction_count, word_count),
        "auxiliary_ratio": _safe_ratio(counts["AUX"], word_count),
        "modal_ratio": _safe_ratio(counts["MODAL"], word_count),
        "function_word_ratio": _safe_ratio(sum(token.pos_ in FUNCTION_POS for token in tokens), word_count),
        "content_word_ratio": _safe_ratio(sum(token.pos_ in CONTENT_POS for token in tokens), word_count),
        "lexical_diversity_root_ttr": _safe_ratio(unique_count, max(word_count, 1) ** 0.5),
        "lexical_diversity_log_ttr": _safe_ratio(unique_count, np.log(max(word_count, 2))),
        "hapax_ratio": _safe_ratio(hapax_count, word_count),
        "long_word_ratio": _safe_ratio(long_count, word_count),
        "repeated_consecutive_word_count": adjacent_repeats,
        "repeated_consecutive_phrase_count": repeated_phrases,
        "filler_disfluency_density": _safe_ratio(filler_count, word_count),
        "clause_like_count": clause_like,
        "clause_like_per_sentence": _safe_ratio(clause_like, sentence_count),
        "conjunction_density": _safe_ratio(conjunction_count, sentence_count),
        "mean_dependency_depth": float(np.mean(dependency_depth)) if dependency_depth else 0.0,
        "mean_sentence_token_count": _safe_ratio(sum(len(list(sentence)) for sentence in sentences), sentence_count),
        "subordinate_clause_count": sum(token.dep_ in {"advcl", "ccomp", "xcomp", "relcl"} for token in tokens),
    }
    return {key: float(value) for key, value in features.items()}


def build_linguistic_v2(transcripts: pd.DataFrame) -> pd.DataFrame:
    nlp = load_nlp()
    rows = [extract_v2_row(text, nlp) for text in transcripts["transcript"].fillna("")]
    extra = pd.DataFrame(rows)
    original = pd.read_csv(OUTPUTS / "linguistic_features.csv")
    original = original[["filename", "label", *ORIGINAL_COLUMNS]]
    result = original.merge(
        pd.concat([transcripts[["filename"]].reset_index(drop=True), extra], axis=1),
        on="filename", how="left", validate="one_to_one",
    )
    return result[["filename", "label", *ORIGINAL_COLUMNS, *extra.columns.tolist()]]


def load_feature_sets() -> dict[str, pd.DataFrame]:
    original = pd.read_csv(OUTPUTS / "linguistic_features.csv")
    v2 = pd.read_csv(OUTPUTS / "linguistic_features_v2.csv")
    acoustic = pd.read_csv(OUTPUTS / "acoustic_features.csv")
    return {"acoustic": acoustic, "original_linguistic": original, "v2_linguistic": v2}


def feature_matrix(feature_set: str, frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    linguistic_name = "v2_linguistic" if "v2" in feature_set else "original_linguistic"
    linguistic = frames[linguistic_name]
    acoustic = frames["acoustic"]
    if feature_set == "acoustic":
        frame = acoustic
    elif feature_set in {"original_linguistic", "v2_linguistic"}:
        frame = linguistic
    elif feature_set == "original_linguistic+acoustic":
        frame = linguistic.merge(acoustic, on=["filename", "label"], validate="one_to_one", suffixes=("", "_acoustic"))
    elif feature_set == "v2_linguistic+acoustic":
        frame = linguistic.merge(acoustic, on=["filename", "label"], validate="one_to_one", suffixes=("", "_acoustic"))
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    x = frame.drop(columns=[column for column in ["filename", "label"] if column in frame])
    return x, frame["label"]