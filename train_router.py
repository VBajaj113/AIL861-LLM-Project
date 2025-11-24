import json
import glob
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# 1. Load all datasets and label them
data = []
labels = []
label_map = {
    # "0": "anxiety",
    # "1": "bipolar",
    # "2": "depression",
    # "3": "ocd",
    # "4": "schizophrenia"
}

files = glob.glob("./data/*_data.json")

for idx, file_path in enumerate(files):
    domain_name = file_path.split("\\")[-1].replace("_data.json", "")
    label_map[idx] = domain_name
    
    with open(file_path, 'r') as f:
        items = json.load(f)
        for item in items:
            data.append(item['instruction']) 
            labels.append(idx)
            data.extend(item['output'].split('.'))
            labels.extend([idx] * len(item['output'].split('.')))

print(f"Loaded {len(data)} training examples across {len(label_map)} domains.")

pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=50000),
    LogisticRegression(class_weight='balanced', max_iter=500)
)

print("Training Orchestrator...")
pipeline.fit(data, labels)

print(label_map.values())
preds = pipeline.predict(data)
print(classification_report(y_true = labels, y_pred = preds, target_names=label_map.values()))
print(label_map)

joblib.dump(pipeline, "models/orchestrator.pkl")
joblib.dump(label_map, "models/label_map.pkl")
print("Orchestrator saved to models/")