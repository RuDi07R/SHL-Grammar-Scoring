from __future__ import annotations

import sys
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))


def extract_original_features(text: str) -> dict[str, float]:
    import re

    text = str(text or "").strip()
    words = re.findall(r"\b[a-zA-Z']+\b", text.lower())
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    lengths = [len(re.findall(r"\b[a-zA-Z']+\b", sentence)) for sentence in sentences]
    fillers = {"um", "uh", "erm", "hmm", "like", "you", "know", "actually", "basically"}
    count = len(words)
    unique = len(set(words))
    return {
        "word_count": count,
        "unique_word_count": unique,
        "sentence_count": len(sentences),
        "avg_sentence_length": float(np.mean(lengths)) if lengths else 0.0,
        "sentence_length_std": float(np.std(lengths)) if lengths else 0.0,
        "type_token_ratio": unique / count if count else 0.0,
        "char_count": len(text),
        "avg_word_length": float(np.mean([len(word) for word in words])) if words else 0.0,
        "repetition_ratio": (count - unique) / count if count else 0.0,
        "filler_count": sum(word in fillers for word in words),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
    }


def extract_acoustic_features(audio_path: Path) -> dict[str, float]:
    y, sr = librosa.load(audio_path, sr=16000)
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    intervals = librosa.effects.split(y, top_db=30, frame_length=2048, hop_length=512)
    speech_duration = np.sum(intervals[:, 1] - intervals[:, 0]) / sr if len(intervals) else 0.0
    pauses = [
        (intervals[i, 0] - intervals[i - 1, 1]) / sr
        for i in range(1, len(intervals))
        if (intervals[i, 0] - intervals[i - 1, 1]) / sr > 0.2
    ]
    f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
    valid_f0 = f0[~np.isnan(f0)]
    return {
        "duration": duration, "speech_ratio": speech_duration / duration if duration else 0.0,
        "silence_ratio": 1 - speech_duration / duration if duration else 1.0,
        "rms_mean": np.mean(rms), "rms_std": np.std(rms), "zcr_mean": np.mean(zcr), "zcr_std": np.std(zcr),
        "pitch_mean": np.mean(valid_f0) if len(valid_f0) else 0.0,
        "pitch_std": np.std(valid_f0) if len(valid_f0) else 0.0,
        "pitch_range": np.max(valid_f0) - np.min(valid_f0) if len(valid_f0) else 0.0,
        "voiced_ratio": np.mean(voiced_flag), "pause_count": len(pauses),
        "pause_mean": np.mean(pauses) if pauses else 0.0, "pause_max": np.max(pauses) if pauses else 0.0,
        "pause_total": np.sum(pauses),
    }


def transcribe_test(test: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    if output_path.exists():
        cached = pd.read_csv(output_path)
    else:
        cached = pd.DataFrame(columns=["filename", "transcript"])
    completed = dict(zip(cached["filename"], cached["transcript"]))
    pending = [name for name in test["filename"] if name not in completed]
    if pending:
        import whisper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper base on {device}; pending={len(pending)}")
        model = whisper.load_model("base", device=device)
        for index, filename in enumerate(pending, start=1):
            result = model.transcribe(str(ROOT / "test" / filename), fp16=False)
            completed[filename] = result["text"].strip()
            pd.DataFrame([{"filename": name, "transcript": completed[name]} for name in test["filename"] if name in completed]).to_csv(output_path, index=False)
            print(f"[{index}/{len(pending)}] {filename}")
    return pd.DataFrame({"filename": test["filename"], "transcript": [completed[name] for name in test["filename"]]})


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    test = pd.read_csv(ROOT / "test.csv")
    transcripts = transcribe_test(test, OUTPUTS / "test_transcripts.csv")
    model = joblib.load(OUTPUTS / "final_model.joblib")
    linguistic = pd.DataFrame([extract_original_features(text) for text in transcripts["transcript"]])
    acoustic = pd.DataFrame([extract_acoustic_features(ROOT / "test" / name) for name in test["filename"]])
    features = pd.concat([linguistic, acoustic], axis=1)
    if model["kind"] == "ensemble":
        x_ridge = features[model["ridge_columns"]]
        x_nonlinear = features[model["nonlinear_columns"]]
        predictions = model["ridge_weight"] * model["ridge"].predict(x_ridge) + (1 - model["ridge_weight"]) * model["nonlinear"].predict(x_nonlinear)
    else:
        predictions = model["pipeline"].predict(features[model["columns"]])
    result = pd.DataFrame({"filename": test["filename"], "prediction": predictions})
    result.to_csv(OUTPUTS / "test_predictions.csv", index=False)
    print(f"Saved {OUTPUTS / 'test_predictions.csv'} rows={len(result)}")


if __name__ == "__main__":
    main()