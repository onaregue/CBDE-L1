"""
G1 - Genera los embeddings de cada frase y los guarda en PostgreSQL con pgvector.

Diferencias con P1 (que usaba REAL[]):
  - La columna es de tipo nativo vector(384). El paquete Python "pgvector" adapta los
    arrays de numpy directamente: no hay conversion a lista ni a literal ARRAY[...].
  - Se pueden crear indices HNSW (uno por metrica, porque la metrica va en la clase de
    operadores del indice). Se crean DESPUES de insertar (mas rapido que insertar con el
    indice ya creado) y su coste se mide aparte: en Chroma el indice se construye dentro
    de add(), asi que para comparar hay que sumar insercion + construccion del indice.

Uso (despues de G0):
    python Pgvector/G1.py

Requisito: pip install pgvector
Opciones (entorno): BATCH_SIZE (500), DEVICE (cpu), METRICS ("l2,cosine"),
                    HNSW_M (16), HNSW_EF_CONSTRUCTION (64)   [valores por defecto de pgvector]
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
DEVICE = os.getenv("DEVICE", "cpu")
METRICS = [m.strip() for m in os.getenv("METRICS", "l2,cosine").split(",")]
HNSW_M = int(os.getenv("HNSW_M", "16"))
HNSW_EF_CONSTRUCTION = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))
MODEL_NAME = "all-MiniLM-L6-v2"
OPCLASS = {"l2": "vector_l2_ops", "cosine": "vector_cosine_ops"}


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(f"No se pudo conectar a PostgreSQL: {e}")


def stats(times):
    a = np.asarray(times)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def print_stats(title, s, total):
    print(f"\n--- {title} ---")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {total:.4f} s")


def main():
    for m in METRICS:
        if m not in OPCLASS:
            sys.exit(f"Metrica no soportada: {m}. Opciones: {', '.join(OPCLASS)}")

    conn = connect()
    cur = conn.cursor()
    try:
        register_vector(conn)
    except psycopg2.ProgrammingError:
        conn.rollback()
        sys.exit("La extension vector no esta activa en esta base: ejecuta primero G0.")

    cur.execute("SELECT id, sentence FROM corpus_g ORDER BY id;")
    rows = cur.fetchall()
    if not rows:
        sys.exit("La tabla corpus_g esta vacia: ejecuta primero G0.")
    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.encode(sentences[:8], show_progress_bar=False)  # calentamiento (no se mide)
    dim = int(model.get_sentence_embedding_dimension())

    cur.execute("DROP TABLE IF EXISTS embeddings_g;")
    cur.execute(f"""
        CREATE TABLE embeddings_g (
            id INT PRIMARY KEY REFERENCES corpus_g(id),
            embedding vector({dim}) NOT NULL
        );
    """)
    conn.commit()

    gen_times, store_times, norms = [], [], []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        texts = sentences[start:start + BATCH_SIZE]

        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False)
        gen_times.append(time.perf_counter() - t0)
        norms.append(np.linalg.norm(embs, axis=1))

        # Los np.ndarray se pasan tal cual: el adaptador de pgvector los serializa.
        t0 = time.perf_counter()
        data = [(ids[start + j], e) for j, e in enumerate(embs)]
        execute_values(cur, "INSERT INTO embeddings_g (id, embedding) VALUES %s",
                       data, template="(%s, %s)", page_size=len(data))
        conn.commit()
        store_times.append(time.perf_counter() - t0)
        n_statements += 1

    # Indices HNSW (uno por metrica), construidos despues de la insercion
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

    norms = np.concatenate(norms)
    gen_s, store_s = stats(gen_times), stats(store_times)
    print(f"\nEmbeddings guardados: {n_rows} (dim {dim}) | sentencias INSERT: {n_statements}")
    print(f"Norma L2 de los vectores: min {norms.min():.4f}, max {norms.max():.4f}")
    print(f"Tamano: tabla {heap_size} | indices HNSW {index_sizes}")
    print_stats(f"[G1] GENERACION DE EMBEDDINGS (por lote de {BATCH_SIZE}, device={DEVICE})",
                gen_s, sum(gen_times))
    print_stats("[G1] ALMACENAMIENTO DE EMBEDDINGS vector(384) (por lote, sin indice, incluye commit)",
                store_s, sum(store_times))
    for m in METRICS:
        print(f"[G1] CONSTRUCCION indice HNSW {m} (m={HNSW_M}, ef_construction="
              f"{HNSW_EF_CONSTRUCTION}): {index_times[m]:.4f} s")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "pgvector_G1.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(), "machine": platform.platform(),
            "model": MODEL_NAME, "device": DEVICE, "batch_size": BATCH_SIZE,
            "metrics": METRICS, "hnsw": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
            "rows": n_rows, "insert_statements": n_statements,
            "norm_min": float(norms.min()), "norm_max": float(norms.max()),
            "table_size": heap_size, "index_sizes": index_sizes,
            "index_build_seconds": index_times,
            "generation_stats_seconds": gen_s, "storage_stats_seconds": store_s,
            "generation_times_seconds": gen_times, "storage_times_seconds": store_times,
        }, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()