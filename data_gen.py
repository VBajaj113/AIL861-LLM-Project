import fitz  
import json
import os
import time
import google.generativeai as genai
import random
from tqdm import tqdm

# --- CONFIG ---
PDF_FOLDER = "./pdfs"
OUTPUT_FOLDER = "./data"
API_KEY = "" #--- YOUR GOOGLE GENAI API KEY HERE ---

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def generate_qa_pairs(text_chunk, domain_label):
    prompt = f"""
    You are an expert in {domain_label}. Read the text below and generate 5 unique Question-Answer pairs.
    Focus on clinical/therapeutic advice found in the text.
    Format strictly as a JSON list: [{{ "instruction": "User Question", "output": "Expert Answer" }}]
    
    TEXT:
    {text_chunk}
    """
    try:
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "")
        return json.loads(clean_text)
    except Exception as e:
        print(f"Error generating: {e}")
        return []


def process_book(filename, domain_label, target_chunks=150):
    print(f"Processing {filename} for domain: {domain_label}...")
    raw_text = extract_text_from_pdf(os.path.join(PDF_FOLDER, filename))
    
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
    
    dataset = []
    output_path = os.path.join(OUTPUT_FOLDER, f"{domain_label}_data.json")
    
    for chunk in tqdm(selected_chunks): 
        qa_pairs = generate_qa_pairs(chunk, domain_label)
        dataset.extend(qa_pairs)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=4)
        time.sleep(2)
        
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=4)
    print(f"SUCCESS: Saved {len(dataset)} pairs to {output_path}")


books_map = {
    "depresion.pdf": "depression",
    "anxity.pdf": "anxiety",
    "ocd.pdf": "ocd",
    "bipolar.pdf": "bipolar",
    "schiz.pdf": "schizophrenia",
}

for book, label in books_map.items():
    process_book(book, label)