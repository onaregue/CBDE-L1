"""
prepare_data.py - Prepara los datos compartidos por PostgreSQL, Chroma y Pgvector.

Genera dos ficheros en la raiz del repositorio:
  - bookcorpus_sentences.json : las primeras 10.000 frases de BookCorpus (el corpus).
                                Solo se descarga si NO existe, para no cambiar el orden:
                                el id de cada frase es su posicion (0..9999).
  - query_sentences.json      : 10 frases de consulta DISTINTAS entre si, elegidas dentro
                                del corpus con semilla fija (42). Lista de
                                {"id": <id en el corpus>, "text": <frase>}.
                                Para cada una se buscan las 2 mas similares entre todas
                                las demas frases del corpus (excluyendo a la propia).

Si query_sentences.json ya existe y es valido, no se toca (borralo para regenerarlo).

Uso:
    python prepare_data.py
"""
import json
import random
from itertools import islice
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SENTENCES_PATH = ROOT / "bookcorpus_sentences.json"
QUERIES_PATH = ROOT / "query_sentences.json"

N_SENTENCES = 10_000
N_QUERIES = 10
SEED = 42
MIN_CHARS = 20  # consultas muy cortas ("yes .") dan embeddings poco informativos

PARQUET_URL = ("https://huggingface.co/datasets/bookcorpus/bookcorpus/resolve/"
               "refs%2Fconvert%2Fparquet/plain_text/train/0000.parquet")


def download_sentences():
    from datasets import load_dataset

    # Tipo 'parquet' explicito para evitar los scripts de carga (.py) no soportados
    ds = load_dataset("parquet", data_files=PARQUET_URL, split="train", streaming=True)
    sentences = [item["text"].strip() for item in islice(ds, N_SENTENCES)]
    with open(SENTENCES_PATH, "w", encoding="utf-8") as f:
        json.dump(sentences, f, ensure_ascii=False, indent=2)
    return sentences


def pick_queries(sentences):
    counts = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    # Texto unico en el corpus (un duplicado tendria una copia exacta a distancia 0
    # como vecino) y no demasiado corto.
    candidates = [i for i, s in enumerate(sentences)
                  if counts[s] == 1 and len(s) >= MIN_CHARS]
    ids = sorted(random.Random(SEED).sample(candidates, N_QUERIES))
    return [{"id": i, "text": sentences[i]} for i in ids]


def queries_are_valid(queries, sentences):
    return (isinstance(queries, list) and len(queries) == N_QUERIES
            and all(isinstance(q, dict) and isinstance(q.get("id"), int)
                    and 0 <= q["id"] < len(sentences)
                    and sentences[q["id"]] == q.get("text") for q in queries)
            and len({q["id"] for q in queries}) == N_QUERIES
            and len({q["text"] for q in queries}) == N_QUERIES)


def main():
    if SENTENCES_PATH.exists():
        with open(SENTENCES_PATH, encoding="utf-8") as f:
            sentences = json.load(f)
        print(f"{SENTENCES_PATH.name} ya existe ({len(sentences)} frases): no se descarga.")
    else:
        sentences = download_sentences()
        print(f"Descargadas {len(sentences)} frases -> {SENTENCES_PATH.name}")

    if QUERIES_PATH.exists():
        with open(QUERIES_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        if queries_are_valid(existing, sentences):
            print(f"{QUERIES_PATH.name} ya existe y es valido: no se toca.")
            return
        print(f"{QUERIES_PATH.name} existe pero no es valido: se regenera.")

    queries = pick_queries(sentences)
    assert queries_are_valid(queries, sentences)
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    print(f"{len(queries)} consultas -> {QUERIES_PATH.name}")
    for q in queries:
        print(f"  [{q['id']}] {q['text'][:80]}")


if __name__ == "__main__":
    main()