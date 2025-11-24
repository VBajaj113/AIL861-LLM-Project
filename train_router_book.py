import fitz
import glob
import joblib
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


# --- CONFIG ---
PDF_FOLDER = "./pdfs"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def process_book(filename, domain_label, target_chunks=50):
    print(f"Processing {filename} for domain: {domain_label}...")
    raw_text = extract_text_from_pdf(os.path.join(PDF_FOLDER, filename))
    
    # Split text into chunks of ~1000 words
    words = raw_text.split()
    all_chunks = [' '.join(words[i:i+1000]) for i in range(0, len(words), 1000)]
    
    print(f"  Found {len(all_chunks)} total chunks.")
    
    # --- SAMPLING LOGIC ---
    # If book is small (like schizophrenia), use all chunks.
    # If book is huge (like bipolar), randomly sample 'target_chunks'.
    if len(all_chunks) > target_chunks:
        selected_chunks = random.sample(all_chunks, target_chunks)
        print(f"  > Randomly sampled {target_chunks} chunks to process.")
    else:
        selected_chunks = all_chunks
        print(f"  > Using all {len(all_chunks)} chunks (book is small).")
    
    return selected_chunks


books_map = {
    "depresion.pdf": "depression",
    "anxity.pdf": "anxiety",
    "ocd.pdf": "ocd",
    "bipolar.pdf": "bipolar",
    "schiz.pdf": "schizophrenia",
}


data = []
labels = []
label_map = {}

files = glob.glob("./data/*_data.json")

for idx, file_path in enumerate(files):
    domain_name = file_path.split("\\")[-1].replace("_data.json", "")
    label_map[idx] = domain_name

    for book, label in books_map.items():
        if label == domain_name:
            chunks = process_book(book, label)

    
    with open(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            all_sentences = [item['instruction']]
            all_sentences.extend(item['output'].split('.'))
            for chunk in chunks:
                all_sentences.extend(chunk.split('.'))
            data.extend(all_sentences)
            labels.extend([idx] * len(all_sentences))

print(f"Loaded {len(data)} training examples across {len(label_map)} domains.")


X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

print(f"Split: {len(X_train)} train / {len(X_test)} test")

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=50000),
    LogisticRegression(class_weight='balanced', max_iter=500, n_jobs=8)
)

print("Training Orchestrator on training split...")
pipeline.fit(X_train, y_train)

ordered_targets = [label_map[i] for i in sorted(label_map.keys())]

print("Evaluation on TEST set:")
preds_test = pipeline.predict(X_test)
print(classification_report(y_true=y_test, y_pred=preds_test, target_names=ordered_targets))

print("Evaluation on TRAIN set (sanity check):")
preds_train = pipeline.predict(X_train)
print(classification_report(y_true=y_train, y_pred=preds_train, target_names=ordered_targets))

print(label_map)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/orchestrator.pkl")
joblib.dump(label_map, "models/label_map.pkl")
print("Orchestrator saved to models/")