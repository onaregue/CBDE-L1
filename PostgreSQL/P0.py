"""
P0 - Load BookCorpus text chunks into PostgreSQL
"""
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
DB_NAME = os.getenv("PGDATABASE")
HOST = os.getenv("PGHOST")
PORT = os.getenv("PGPORT")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))


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
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    conn = connect()
    cur = conn.cursor()
    cur.execute("SHOW server_version;")
    pg_version = cur.fetchone()[0]

    # Simple relational schema. ID corresponds to the sentence position (0..N-1)
    cur.execute("DROP TABLE IF EXISTS embeddings; DROP TABLE IF EXISTS corpus;")
    cur.execute("CREATE TABLE corpus (id INT PRIMARY KEY, sentence TEXT NOT NULL);")
    conn.commit()

    times = []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[start:start + BATCH_SIZE]
        rows = [(start + j, s) for j, s in enumerate(chunk)]

        t0 = time.perf_counter()
        
        # Single multi-row INSERT per batch
        execute_values(cur, "INSERT INTO corpus (id, sentence) VALUES %s",
                       rows, page_size=len(rows))
        conn.commit()
        times.append(time.perf_counter() - t0)
        n_statements += 1

    cur.execute("SELECT count(*) FROM corpus;")
    n_rows = cur.fetchone()[0]
    cur.close()
    conn.close()

    s = stats(times)
    print("\n--- [P0] TEXT INSERTION (per batch, includes commit) ---")
    print(f"Inserted rows: {n_rows} | Batches: {len(times)} of {BATCH_SIZE} | INSERT statements: {n_statements}")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Mean: {s['mean']:.6f} s")
    print(f"Standard deviation: {s['std']:.6f} s")
    print(f"Total time: {sum(times):.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "postgres_P0.json", "w", encoding="utf-8") as f:
        json.dump({
            "postgres_version": pg_version,
            "python": platform.python_version(),
            "machine": platform.platform(),
            "batch_size": BATCH_SIZE,
            "rows": n_rows,
            "insert_statements": n_statements,
            "stats_seconds": s,
            "batch_times_seconds": times
        }, f, indent=2)


if __name__ == "__main__":
    main()