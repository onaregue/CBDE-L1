"""
P1 - Genera los embeddings de cada frase y los guarda en PostgreSQL (sin pgvector).

Uso (despues de P0):
    python PostgreSQL/P1.py

Conexion: mismas variables de entorno que P0 (PGDATABASE, PGHOST, PGPORT, PGUSER, PGPASSWORD).
Otras opciones: BATCH_SIZE (500), DEVICE (por defecto cpu, para que los tiempos sean
comparables entre ordenadores y entre PostgreSQL y Chroma).
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
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_NAME = "all-MiniLM-L6-v2"


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(
            f"No se pudo conectar a PostgreSQL: {e}\n"
            "Comprueba que el servidor esta arrancado, que has ejecutado P0 y, "
            "si hace falta, define PGUSER / PGPASSWORD / PGHOST / PGPORT."
        )


def stats(times):
    a = np.asarray(times)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def save_results(name, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_stats(title, s, total):
    print(f"\n--- {title} ---")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {total:.4f} s")


def main():
    conn = connect()
    cur = conn.cursor()

    cur.execute("SELECT id, sentence FROM corpus ORDER BY id;")
    rows = cur.fetchall()
    if not rows:
        sys.exit("La tabla corpus esta vacia: ejecuta primero P0.")
    ids = [r[0] for r in rows]
    sentences = [r[1] for r in rows]

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
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

    gen_times, store_times, norms = [], [], []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        texts = sentences[start:start + BATCH_SIZE]

        # 1) Generacion del embedding (medida por separado del almacenamiento)
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False)
        gen_times.append(time.perf_counter() - t0)
        norms.append(np.linalg.norm(embs, axis=1))

        # 2) Almacenamiento. Convertir numpy -> lista Python -> literal ARRAY[...] es
        #    parte del coste del desajuste de impedancia, por eso entra en el tiempo.
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

    norms = np.concatenate(norms)
    gen_s, store_s = stats(gen_times), stats(store_times)
    print(f"\nEmbeddings guardados: {n_rows} (dim {len(data[0][1])}) | "
          f"sentencias INSERT: {n_statements}")
    print(f"Norma L2 de los vectores: min {norms.min():.4f}, max {norms.max():.4f} "
          "(~1 => normalizados: coseno, L2 y producto escalar dan el mismo ranking)")
    print(f"Tamano tabla embeddings: {heap_size} (heap) / {total_size} (con TOAST e indice)")
    print_stats(f"[P1] GENERACION DE EMBEDDINGS (por lote de {BATCH_SIZE}, device={DEVICE})",
                gen_s, sum(gen_times))
    print_stats("[P1] ALMACENAMIENTO DE EMBEDDINGS (por lote, incluye commit)",
                store_s, sum(store_times))

    save_results("postgres_P1.json", {
        "python": platform.python_version(), "machine": platform.platform(),
        "model": MODEL_NAME, "device": DEVICE, "batch_size": BATCH_SIZE,
        "rows": n_rows, "insert_statements": n_statements,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "table_size": {"heap": heap_size, "total": total_size},
        "generation_stats_seconds": gen_s, "storage_stats_seconds": store_s,
        "generation_times_seconds": gen_times, "storage_times_seconds": store_times,
    })


if __name__ == "__main__":
    main()