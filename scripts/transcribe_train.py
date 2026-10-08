import os
import time
import pandas as pd
import whisper

TRAIN_CSV = "train.csv"
AUDIO_DIR = "train"
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "train_transcripts.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

train = pd.read_csv(TRAIN_CSV)

if os.path.exists(OUTPUT_FILE):
    results = pd.read_csv(OUTPUT_FILE)
    completed = set(results["filename"])
    print(f"Resuming. Already completed: {len(completed)}")
else:
    results = pd.DataFrame(columns=["filename", "label", "transcript"])
    completed = set()

print("Loading Whisper...")
model = whisper.load_model("base", device="cuda")
print("Whisper loaded on GPU.")

start_total = time.time()

for i, row in train.iterrows():
    filename = row["filename"]

    if filename in completed:
        continue

    audio_path = os.path.join(AUDIO_DIR, filename)

    start = time.time()

    try:
        result = model.transcribe(
            audio_path,
            fp16=False
        )

        transcript = result["text"].strip()

        new_row = pd.DataFrame([{
            "filename": filename,
            "label": row["label"],
            "transcript": transcript
        }])

        results = pd.concat([results, new_row], ignore_index=True)
        results.to_csv(OUTPUT_FILE, index=False)

        elapsed = time.time() - start

        print(
            f"[{len(results)}/{len(train)}] "
            f"{filename} | {elapsed:.2f}s"
        )

    except Exception as e:
        print(f"ERROR: {filename} -> {e}")

total_time = time.time() - start_total

print("\nFinished.")
print(f"Transcripts saved to: {OUTPUT_FILE}")
print(f"Total time: {total_time / 60:.2f} minutes")