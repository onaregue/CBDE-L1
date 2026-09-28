"""
P1 - Generate and store sentence embeddings in PostgreSQL as REAL[].
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
DB_NAME = os.getenv("PGDATABASE")
HOST = os.getenv("PGHOST") 
PORT = os.getenv("PGPORT")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
MODEL_NAME = "all-MiniLM-L6-v2"


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST, port=PORT, user=USER, password=PASSWORD)
    except psycopg2.OperationalError as e:
        sys.exit(f"Error connecting to PostgreSQL: {e}")


def stats(times):
    a = np.asarray(times)
    return {
        "min": float(a.min()), 
        "max": float(a.max()),
        "mean": float(a.mean()), 
        "std": float(a.std())
    }

def main():
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT id, sentence FROM corpus ORDER BY id;")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    model.encode(sentences[:8], show_progress_bar=False)  # calentamiento (no se mide)

    # Tabla aparte (INSERT en lugar de UPDATE: un UPDATE reescribe la fila entera y
    # deja tuplas muertas). REAL[] = float32, igual que el modelo, la mitad de espacio
    # que FLOAT8[]. El vector es, para PostgreSQL, un array opaco: no hay indice ni
    # operador de similitud nativo.
    cur.execute("DROP TABLE IF EXISTS embeddings;")
    cur.execute("""
        CREATE TABLE embeddings (
            id INT PRIMARY KEY REFERENCES corpus(id),
            embedding REAL[] NOT NULL
        );
    """)
    conn.commit()

    gen_times, store_times = [], []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        texts = sentences[start:start + BATCH_SIZE]

        # generate embeddings
        t0 = time.perf_counter()

        embs = model.encode(texts, batch_size=64, show_progress_bar=False)
        gen_times.append(time.perf_counter() - t0)

        # storage numpy -> lista Python -> literal ARRAY[...] 
        t0 = time.perf_counter()

        data = [(ids[start + j], e.tolist()) for j, e in enumerate(embs)]
        execute_values(cur, "INSERT INTO embeddings (id, embedding) VALUES %s",
                       data, template="(%s, %s::real[])", page_size=len(data))
        conn.commit()
        store_times.append(time.perf_counter() - t0)
        n_statements += 1

    cur.execute("ANALYZE embeddings;")
    conn.commit()
    cur.execute("SELECT count(*) FROM embeddings;")
    n_rows = cur.fetchone()[0]
    cur.execute("""SELECT pg_size_pretty(pg_relation_size('embeddings')),
                          pg_size_pretty(pg_total_relation_size('embeddings'));""")
    heap_size, total_size = cur.fetchone()
    cur.close()
    conn.close()

    gen_s, store_s = stats(gen_times), stats(store_times)
    print(f"\nStored embeddings: {n_rows} (dim {len(data[0][1])}) | INSERT statements: {n_statements}")
    print(f"Table size: {heap_size} (heap) / {total_size} (total with toast and indexes)")

    print(f"\n--- [P1] EMBEDDING GENERATION (per batch of {BATCH_SIZE}, device=cpu) ---")
    print(f"Min:  {gen_s['min']:.6f} s")
    print(f"Max:  {gen_s['max']:.6f} s")
    print(f"Mean: {gen_s['mean']:.6f} s")
    print(f"Standard deviation: {gen_s['std']:.6f} s")
    print(f"Total time: {sum(gen_times):.4f} s")

    print(f"\n--- [P1] EMBEDDING STORAGE (per batch, includes commit) ---")
    print(f"Min:  {store_s['min']:.6f} s")
    print(f"Max:  {store_s['max']:.6f} s")
    print(f"Mean: {store_s['mean']:.6f} s")
    print(f"Standard deviation: {store_s['std']:.6f} s")
    print(f"Total time: {sum(store_times):.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "postgres_P1.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(),
            "machine": platform.platform(),
            "model": MODEL_NAME,
            "device": "cpu",
            "batch_size": BATCH_SIZE,
            "rows": n_rows,
            "insert_statements": n_statements,
            "table_size": {"heap": heap_size, "total": total_size},
            "generation_stats_seconds": gen_s,
            "storage_stats_seconds": store_s,
            "generation_times_seconds": gen_times,
            "storage_times_seconds": store_times
        }, f, indent=2)


if __name__ == "__main__":
    main()