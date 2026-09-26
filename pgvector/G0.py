"""
G0 - Carga del texto de BookCorpus en PostgreSQL con la extension pgvector.

Igual que P0, pero: (1) activa la extension "vector" y (2) usa tablas propias
(corpus_g, embeddings_g) para no pisar las tablas de P0/P1 en la misma base.

Uso (despues de prepare_data.py):
    python Pgvector/G0.py

Requisitos: pgvector instalado en el servidor (brew install pgvector, o compilarlo desde
https://github.com/pgvector/pgvector) y la base creada (createdb cbde_l1).
Conexion: PGDATABASE (cbde_l1), PGHOST (localhost), PGPORT, PGUSER, PGPASSWORD.
Otras opciones: BATCH_SIZE (500).
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(
            f"No se pudo conectar a PostgreSQL: {e}\n"
            f"Comprueba que el servidor esta arrancado y que la base existe (createdb {DB_NAME})."
        )


def stats(times):
    a = np.asarray(times)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def main():
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    conn = connect()
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
    except psycopg2.Error as e:
        conn.rollback()
        sys.exit(f"No se pudo activar la extension vector: {e}\n"
                 "Instala pgvector en el servidor PostgreSQL (ver cabecera del script).")
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    pgvector_version = cur.fetchone()[0]
    cur.execute("SHOW server_version;")
    pg_version = cur.fetchone()[0]

    cur.execute("DROP TABLE IF EXISTS embeddings_g; DROP TABLE IF EXISTS corpus_g;")
    cur.execute("CREATE TABLE corpus_g (id INT PRIMARY KEY, sentence TEXT NOT NULL);")
    conn.commit()

    times = []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        rows = [(start + j, s) for j, s in enumerate(chunk)]
        execute_values(cur, "INSERT INTO corpus_g (id, sentence) VALUES %s",
                       rows, page_size=len(rows))
        conn.commit()
        times.append(time.perf_counter() - t0)
        n_statements += 1

    cur.execute("SELECT count(*) FROM corpus_g;")
    n_rows = cur.fetchone()[0]
    cur.close()
    conn.close()

    s = stats(times)
    print(f"\n--- [G0] INSERCION DEL TEXTO (por lote, incluye commit) ---")
    print(f"PostgreSQL {pg_version} | pgvector {pgvector_version}")
    print(f"Filas insertadas: {n_rows} | lotes: {len(times)} de {BATCH_SIZE} "
          f"| sentencias INSERT: {n_statements}")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {sum(times):.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "pgvector_G0.json", "w", encoding="utf-8") as f:
        json.dump({"postgres_version": pg_version, "pgvector_version": pgvector_version,
                   "python": platform.python_version(), "machine": platform.platform(),
                   "batch_size": BATCH_SIZE, "rows": n_rows,
                   "insert_statements": n_statements, "stats_seconds": s,
                   "batch_times_seconds": times}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()