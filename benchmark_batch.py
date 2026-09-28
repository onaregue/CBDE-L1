"""
benchmark_batch_size.py - Test ultra-rapid per trobar el batch size optim (5 segons).
"""
import json
import os
import time
from pathlib import Path
import chromadb
from chromadb.config import Settings
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
PORT = os.getenv("PGPORT", "5432")
USER = os.getenv("PGUSER", "postgres")
PASSWORD = os.getenv("PGPASSWORD", "postgres")

# Provarem aquests batch sizes sobre una mostra de 2.000 frases
CANDIDATES = [100, 500, 1000, 2000]
SAMPLE_SIZE = 2000
DUMMY_VEC = [0.01] * 384


def test_postgres(sentences, batch_size):
    conn = psycopg2.connect(dbname=DB_NAME, host=HOST, port=PORT, user=USER, password=PASSWORD)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS test_batch; CREATE TABLE test_batch (id INT PRIMARY KEY, s TEXT);")
    conn.commit()

    t0 = time.perf_counter()
    for start in range(0, len(sentences), batch_size):
        chunk = [(start + j, s) for j, s in enumerate(sentences[start:start + batch_size])]
        execute_values(cur, "INSERT INTO test_batch (id, s) VALUES %s", chunk, page_size=len(chunk))
        conn.commit()
    total_time = time.perf_counter() - t0

    cur.execute("DROP TABLE test_batch;")
    conn.commit()
    cur.close()
    conn.close()
    return total_time


def test_chroma(sentences, batch_size):
    client = chromadb.PersistentClient(path="./chroma_test", settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection("test_col")
    except Exception:
        pass
    col = client.create_collection("test_col", metadata={"hnsw:space": "l2"}, embedding_function=None)

    t0 = time.perf_counter()
    for start in range(0, len(sentences), batch_size):
        chunk = sentences[start:start + batch_size]
        ids = [str(start + j) for j in range(len(chunk))]
        embs = [DUMMY_VEC] * len(chunk)
        col.add(ids=ids, documents=chunk, embeddings=embs)
    total_time = time.perf_counter() - t0

    try:
        client.delete_collection("test_col")
    except Exception:
        pass
    return total_time


def main():
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)[:SAMPLE_SIZE]

    print(f"Executant test de Batch Sizes sobre {len(sentences)} frases...")
    print(f"{'Batch Size':<12} | {'PostgreSQL (s)':<16} | {'Chroma (s)':<16}")
    print("-" * 50)

    results = []
    for b in CANDIDATES:
        t_pg = test_postgres(sentences, b)
        t_ch = test_chroma(sentences, b)
        results.append({"batch_size": b, "postgres_seconds": t_pg, "chroma_seconds": t_ch})
        print(f"{b:<12} | {t_pg:<16.4f} | {t_ch:<16.4f}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "batch_size_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nTest completat en pocs segons!")


if __name__ == "__main__":
    main()