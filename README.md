# SHL Grammar Scoring Engine

This project estimates a continuous 1-5 grammar score for spoken-English assessment recordings. It was developed for a private SHL hiring challenge. The public repository contains the reusable code, notebooks, and documentation only; the SHL-provided dataset and audio are intentionally excluded.

## Problem and approach

The task is supervised regression from spoken responses to human grammar scores. The pipeline is:

```text
audio -> Whisper base transcription -> linguistic + acoustic features
			-> Ridge and HistGradientBoosting -> 50/50 ensemble -> grammar score
```

Whisper `base` converts each recording into text. Test transcription uses `fp16=False` and CUDA when available. Missing transcript values are retained for row alignment and treated as empty text during feature extraction.

### Preprocessing and feature engineering

- Original linguistic features cover word and character counts, unique words, sentence counts and lengths, lexical diversity, average word length, repetition, fillers, questions, and exclamations.
- Acoustic features cover duration, speech/silence ratio, RMS energy, zero-crossing rate, pitch statistics, voiced ratio, and pause statistics.
- Missing numeric values are median-imputed inside the scikit-learn pipelines. Ridge features are standardized inside the pipeline.
- An expanded V2 linguistic representation adds POS, lexical diversity, repetition, disfluency, dependency, clause, and sentence-complexity features. It is evaluated as an experiment and is not the selected final feature set.

## Model selection and results

The experiment protocol uses `KFold(n_splits=5, shuffle=True, random_state=42)`. Ridge regularization values include `alpha=30`, and conservative nonlinear regressors are compared on the strongest feature sets.

The selected model uses original linguistic plus acoustic features:

- Ridge Regression with `alpha=30`
- HistGradientBoosting
- 50/50 prediction weighting: 0.5 Ridge and 0.5 HistGradientBoosting

| Model | CV RMSE | CV Pearson |
| --- | ---: | ---: |
| Ridge, alpha=30 | 0.8151173374725303 | 0.7527653760330594 |
| HistGradientBoosting | 0.7898459026180773 | 0.7715411286723808 |
| 50/50 ensemble | 0.7731985964727636 | 0.7812503393565036 |

The final 5-fold CV RMSE is `0.7731985964727636` and the final 5-fold CV Pearson correlation is `0.7812503393565036`. Notebook 13 also computes the compulsory in-sample training RMSE from the saved fitted artifact; it is a training diagnostic and must not be confused with cross-validation performance.

## Repository structure

```text
README.md
requirements.txt
notebooks/01_data_inspection.ipynb ... 14_final_submission.ipynb
scripts/
	build_linguistic_v2.py
	pipeline_common.py
	predict_test.py
	run_experiments.py
	transcribe_train.py
```

The SHL challenge dataset and audio files were provided privately for the hiring assessment and are intentionally not included in this public repository. The private `train/`, `test/`, CSV files, WAV files, generated `outputs/`, and final submission file are excluded by `.gitignore` and are not part of this public repository.

## Reproduction with authorized private data

After obtaining the challenge data through the authorized SHL process, place the local files at the project root as `train/`, `test/`, `train.csv`, and `test.csv`. Keep these files local and do not commit them.

Create an environment and install the dependencies from `requirements.txt`. The V2 pipeline also requires the spaCy model `en_core_web_sm` to be available locally. Then run from the project root:

```powershell
python .\scripts\transcribe_train.py
python .\scripts\build_linguistic_v2.py
python .\scripts\run_experiments.py
python .\scripts\predict_test.py
```

The experiment script writes private artifacts under `outputs/`, including the selected model. The prediction script uses the current `test.csv` filenames, resumes cached test transcription when available, and writes private prediction files under `outputs/`. Notebook 14 validates the predictions and can create `outputs/submission.csv` for the authorized submission process. That submission file must not be published.

## Privacy

The private SHL challenge dataset, audio files, test data, predictions, cached transcripts, model artifacts, and submission files are intentionally excluded from this public repository. Do not add credentials, API keys, or private challenge artifacts to GitHub.
