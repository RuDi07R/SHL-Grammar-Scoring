# SHL Grammar Scoring Engine

A machine-learning pipeline for predicting the **grammar quality of spoken English** from short audio recordings.

This project was developed for a private **SHL hiring assessment**. The goal was to predict a continuous grammar score from **1 to 5**, where higher scores indicate better grammatical performance.

> **Important:** The SHL dataset, audio recordings, test data, generated predictions, and other private challenge artifacts are intentionally **not included** in this public repository.

---

## 1. What problem does this solve?

For each spoken-English response, we are given an audio recording and a human-assigned grammar score.

The system learns the relationship between the audio and those scores, then predicts a grammar score for an unseen recording.

The main challenge is that **grammar is not directly present in an audio signal**. A useful system therefore needs to understand both:

1. **What the speaker said** — words, vocabulary, sentences, repetition, fillers, etc.
2. **How the speaker spoke** — pauses, pitch, energy, speech/silence patterns, etc.

The final system combines both sources of information.

---

## 2. Overall approach

The complete pipeline can be summarized as:

```text
                 Audio Recording
                       |
                       v
                Whisper (base)
                       |
                       v
                  Transcript
                       |
          +------------+------------+
          |                         |
          v                         v
   Linguistic Features       Acoustic Features
          |                         |
          +------------+------------+
                       |
                       v
                Feature Matrix
                       |
             +---------+---------+
             |                   |
             v                   v
       Ridge Regression   HistGradientBoosting
             |                   |
             +---------+---------+
                       |
                       v
                 50/50 Ensemble
                       |
                       v
              Predicted Score (1-5)
```

### Why this design?

Instead of relying on a single type of feature, the model uses complementary information.

- **Whisper** gives us a transcript of the spoken response.
- **Linguistic features** describe the language used by the speaker.
- **Acoustic features** describe speech characteristics such as pauses and pitch.
- **Ridge Regression** provides a strong, stable linear model.
- **HistGradientBoosting** can capture nonlinear relationships between features.
- Their predictions are combined using a **50/50 ensemble**.

---

## 3. Speech-to-text with Whisper

The first step is converting every recording into text.

I used **OpenAI Whisper base** for transcription.

```text
Audio -> Whisper -> Transcript
```

The transcription is then used to calculate linguistic features.

For GPU inference, CUDA is used when available. Whisper inference was configured with `fp16=False` for compatibility with the available GPU.

Missing transcripts are kept in the dataset so that row alignment is preserved. They are treated as empty text during linguistic feature extraction.

---

## 4. Feature engineering

The final model uses two main groups of features.

### A. Linguistic features

These features describe the actual language contained in the transcript.

Examples include:

| Feature | What it represents |
| --- | --- |
| Word count | Amount of speech produced |
| Unique word count | Vocabulary variety |
| Character count | Overall transcript length |
| Sentence count | Sentence structure |
| Average sentence length | Typical sentence complexity |
| Sentence length variation | Variation in sentence construction |
| Type-token ratio | Lexical diversity |
| Average word length | Vocabulary/word structure |
| Repetition ratio | Repeated language |
| Filler count | Disfluencies such as fillers |
| Question count | Question usage |
| Exclamation count | Exclamation usage |

These features help the model estimate how varied, structured, and complex the speaker's language is.

### B. Acoustic features

The audio itself also contains useful information.

The acoustic feature set includes:

- Recording duration
- Speech ratio
- Silence ratio
- RMS energy
- Energy variation
- Zero-crossing rate
- Pitch mean
- Pitch variation
- Pitch range
- Voiced ratio
- Number of pauses
- Average pause duration
- Maximum pause duration
- Total pause duration

For example, the number and duration of pauses can provide information about speaking behavior that cannot be obtained from text alone.

---

## 5. Why combine linguistic and acoustic information?

This was one of the most important findings during experimentation.

A transcript tells us **what was said**, but it does not capture everything about the recording.

For example, two speakers could produce similar sentences while having very different:

- pause patterns
- speech/silence ratios
- pitch variation
- energy variation
- speaking rhythm

Conversely, acoustic information alone cannot tell us about vocabulary or sentence construction.

Combining the two gives the model a broader representation of the response.

---

## 6. Model selection

Several approaches were tested rather than assuming that a more complicated model would automatically perform better.

The main evaluation used:

```text
5-Fold Cross Validation
KFold(n_splits=5, shuffle=True, random_state=42)
```

The evaluation metrics were the same two metrics used for the challenge:

- **RMSE** — lower is better
- **Pearson correlation** — higher is better

### Main results

| Model | CV RMSE ↓ | CV Pearson ↑ |
| --- | ---: | ---: |
| Ridge Regression (alpha=30) | 0.8151 | 0.7528 |
| HistGradientBoosting | 0.7898 | 0.7715 |
| **50/50 Ensemble** | **0.7732** | **0.7813** |

The ensemble combines:

```text
Final prediction =
0.5 × Ridge prediction
+
0.5 × HistGradientBoosting prediction
```

The ensemble produced the strongest validation result among the evaluated final candidates.

---

## 7. Final model

The selected feature set is:

```text
Original Linguistic Features
            +
Acoustic Features
```

The final model consists of:

### Ridge Regression

Ridge was used with:

```text
alpha = 30
```

It provides a strong regularized linear baseline and helps prevent unstable coefficients when features are correlated.

### HistGradientBoosting

HistGradientBoosting was selected as the nonlinear component because it can model relationships that a purely linear model may miss.

### Ensemble

The two models are combined equally:

```text
50% Ridge
+
50% HistGradientBoosting
```

This gave a better validation result than either component alone.

---

## 8. Training diagnostic

The challenge also requires reporting training-data RMSE.

The fitted final model produced:

```text
Training RMSE: 0.5480
```

This is an **in-sample training diagnostic**. It is not used as the main estimate of generalization performance.

The more important performance numbers are the 5-fold cross-validation results:

```text
CV RMSE:     0.7732
CV Pearson:  0.7813
```

---

## 9. Additional experiments

The project also includes experiments with:

### Text embeddings

Sentence embeddings using **all-MiniLM-L6-v2** were tested.

They were useful as an experiment, but did not outperform the simpler engineered linguistic + acoustic representation.

### Expanded linguistic features

A second linguistic representation was also developed containing additional:

- Part-of-speech features
- Lexical diversity measures
- Repetition features
- Disfluency measures
- Dependency features
- Clause-related features
- Sentence-complexity features

These experiments are documented in the notebooks, but the original linguistic + acoustic feature set was selected for the final model.

This is intentional: the final system was chosen based on validation performance rather than simply adding more features.

---

## 10. Repository structure

The notebooks are organized as a progression from data understanding to the final submission pipeline.

```text
SHL-Grammar-Scoring/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_data_inspection.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_whisper_test.ipynb
│   ├── 04_transcript_inspection.ipynb
│   ├── 05_linguistic_features.ipynb
│   ├── 06_linguistic_model.ipynb
│   ├── 07_text_embedding.ipynb
│   ├── 08_embedding_model.ipynb
│   ├── 09_feature_fusion.ipynb
│   ├── 10_acoustic_features.ipynb
│   ├── 11_linguistic_features_v2.ipynb
│   ├── 12_feature_experiments.ipynb
│   ├── 13_final_model.ipynb
│   └── 14_final_submission.ipynb
│
└── scripts/
    ├── build_linguistic_v2.py
    ├── pipeline_common.py
    ├── predict_test.py
    ├── run_experiments.py
    └── transcribe_train.py
```

### Notebook roadmap

| Notebook | Purpose |
| --- | --- |
| 01 | Inspect the dataset and understand the task |
| 02 | Establish a simple baseline |
| 03–04 | Test Whisper and inspect generated transcripts |
| 05–06 | Build linguistic features and evaluate a first model |
| 07–08 | Experiment with text embeddings |
| 09 | Combine feature representations |
| 10 | Extract and evaluate acoustic features |
| 11 | Build expanded linguistic features |
| 12 | Compare models and select the final approach |
| 13 | Final model evaluation and diagnostics |
| 14 | Test prediction and submission validation |

This structure makes it possible to follow **why the final model was selected**, rather than only seeing the final result.

---

## 11. Reproducing the pipeline

The public repository does not contain the private SHL data.

With authorized access to the challenge data, the expected local structure is:

```text
train/
test/
train.csv
test.csv
```

The private files should remain local.

Install the dependencies from `requirements.txt`, make sure the spaCy model `en_core_web_sm` is available, and run from the project root:

```powershell
python .\scripts\transcribe_train.py
python .\scripts\build_linguistic_v2.py
python .\scripts\run_experiments.py
python .\scripts\predict_test.py
```

The scripts generate private artifacts under `outputs/`.

Notebook 14 performs the final prediction/submission validation.

---

## 12. Privacy and challenge data

This repository is intentionally limited to the reusable implementation and documentation.

The following are **not included**:

- SHL audio recordings
- Training/test datasets
- Private CSV files
- Transcripts generated from the private data
- Trained model artifacts
- Prediction files
- Submission files
- API keys or credentials

These files are also excluded through `.gitignore`.

The purpose is to make the methodology and implementation publicly understandable without exposing private hiring-assessment data.

---

## 13. Key takeaway

The final approach is deliberately straightforward:

```text
Speech
  ↓
Whisper transcription
  ↓
Linguistic + acoustic feature engineering
  ↓
Ridge + HistGradientBoosting
  ↓
50/50 ensemble
  ↓
Grammar score from 1–5
```

The best validation result was:

**RMSE = 0.7732**  
**Pearson correlation = 0.7813**

The main insight from the experiments was that **combining information from the transcript and the audio signal worked better than relying on either representation alone**.

---

## License / Challenge Notice

This repository contains work developed for a private SHL hiring assessment. The challenge dataset and audio remain private and are not redistributed here.
