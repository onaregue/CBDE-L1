"""
G1 - Generate embeddings, store in native vector(384), and build HNSW indexes
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
from pgvector.psycopg2 import register_vector
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
METRICS = ["l2", "cosine"]
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64
OPCLASS = {"l2": "vector_l2_ops", "cosine": "vector_cosine_ops"}

def stats(times):
    a = np.asarray(times)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std())
    }

def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(f"Error connecting to PostgreSQL: {e}")

def main():

    conn = connect()
    cur = conn.cursor()
    register_vector(conn)

    cur.execute("SELECT id, sentence FROM corpus_g ORDER BY id;")
    rows = cur.fetchall()
    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device="cpu")
    dim = int(model.get_embedding_dimension())

    cur.execute("DROP TABLE IF EXISTS embeddings_g;")
    cur.execute(f"""
        CREATE TABLE embeddings_g (
            id INT PRIMARY KEY REFERENCES corpus_g(id),
            embedding vector({dim}) NOT NULL
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

        # store embeddings in the database
        t0 = time.perf_counter()
        data = [(ids[start + j], e) for j, e in enumerate(embs)]
        execute_values(cur, "INSERT INTO embeddings_g (id, embedding) VALUES %s",
                       data, template="(%s, %s)", page_size=len(data))
        conn.commit()
        store_times.append(time.perf_counter() - t0)
        n_statements += 1

    # index construction
    index_times, index_sizes = {}, {}
    for m in METRICS:
        t0 = time.perf_counter()
        cur.execute(f"CREATE INDEX embeddings_g_hnsw_{m} ON embeddings_g "
                    f"USING hnsw (embedding {OPCLASS[m]}) "
                    f"WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});")
        conn.commit()
        index_times[m] = time.perf_counter() - t0

    cur.execute("ANALYZE embeddings_g;")
    conn.commit()
    cur.execute("SELECT count(*) FROM embeddings_g;")
    n_rows = cur.fetchone()[0]
    cur.execute("SELECT pg_size_pretty(pg_relation_size('embeddings_g'));")
    heap_size = cur.fetchone()[0]
    for m in METRICS:
        cur.execute(f"SELECT pg_size_pretty(pg_relation_size('embeddings_g_hnsw_{m}'));")
        index_sizes[m] = cur.fetchone()[0]
    cur.close()
    conn.close()

    gen_s, store_s = stats(gen_times), stats(store_times)
    print(f"\nStored embeddings: {n_rows} (dim {dim}) | INSERT statements: {n_statements}")
    print(f"Table size: {heap_size} | HNSW Index sizes: {index_sizes}")

    print(f"\n--- [G1] EMBEDDING GENERATION (per batch of {BATCH_SIZE}, device=cpu) ---")
    print(f"Min:  {gen_s['min']:.6f} s")
    print(f"Max:  {gen_s['max']:.6f} s")
    print(f"Mean: {gen_s['mean']:.6f} s")
    print(f"Standard deviation: {gen_s['std']:.6f} s")
    print(f"Total time: {sum(gen_times):.4f} s")

    print(f"\n--- [G1] STORAGE vector({dim}) (per batch, raw table without index) ---")
    print(f"Min:  {store_s['min']:.6f} s")
    print(f"Max:  {store_s['max']:.6f} s")
    print(f"Mean: {store_s['mean']:.6f} s")
    print(f"Standard deviation: {store_s['std']:.6f} s")
    print(f"Total time: {sum(store_times):.4f} s")

    for m in METRICS:
        print(f"\n--- [G1] HNSW INDEX BUILD ({m.upper()}, m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}) ---")
        print(f"Build time: {index_times[m]:.4f} s | Index disk size: {index_sizes[m]}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "pgvector_G1.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(),
            "machine": platform.platform(),
            "model": MODEL_NAME,
            "device": "cpu",
            "batch_size": BATCH_SIZE,
            "metrics": METRICS,
            "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            "rows": n_rows,
            "insert_statements": n_statements,
            "table_size": heap_size,
            "index_sizes": index_sizes,
            "index_build_seconds": index_times,
            "generation_stats_seconds": gen_s,
            "storage_stats_seconds": store_s,
            "generation_times_seconds": gen_times,
            "storage_times_seconds": store_times
        }, f, indent=2)



if __name__ == "__main__":
    main()