import json
from datasets import load_dataset

parquet_url = "https://huggingface.co/datasets/bookcorpus/bookcorpus/resolve/refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet"

# Indiquem expressament el tipus 'parquet' per evitar que busqui scripts .py
bookCorpus = load_dataset("parquet", data_files=parquet_url, split="train", streaming=True)
sentences = []

# mirar de agafar frases que no siguin les primeres¿
for item in bookCorpus:
    sentences.append(item["text"].strip())
    if len(sentences) >= 10000:
        break

with open("bookcorpus_sentences.json", "w") as f:
    json.dump(sentences, f, ensure_ascii=False, indent=2)

query_sentences = sentences[:10]
with open("query_sentences.json", "w") as f:
    json.dump(query_sentences, f, ensure_ascii=False, indent=2)