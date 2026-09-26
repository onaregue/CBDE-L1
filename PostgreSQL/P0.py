"""
P0 - Carga del texto de BookCorpus en PostgreSQL (sin pgvector).

Uso (desde cualquier carpeta):
    python PostgreSQL/P0.py

Conexion (variables de entorno estandar de PostgreSQL, todas opcionales):
    PGDATABASE (por defecto cbde_l1), PGHOST (localhost), PGPORT, PGUSER, PGPASSWORD
Si no defines PGUSER, se usa el usuario del sistema (caso de Homebrew en macOS).
La base de datos debe existir ya (createdb cbde_l1).

Otras opciones: BATCH_SIZE (por defecto 500).
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
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2000"))


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(
            f"No se pudo conectar a PostgreSQL: {e}\n"
            f"Comprueba que el servidor esta arrancado, que la base existe "
            f"(createdb {DB_NAME}) y, si hace falta, define PGUSER / PGPASSWORD / "
            "PGHOST / PGPORT."
        )


def stats(times):
    a = np.asarray(times)
    # std poblacional (ddof=0): usar la misma en todos los scripts y en Chroma
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def save_results(name, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    conn = connect()
    cur = conn.cursor()
    cur.execute("SHOW server_version;")
    pg_version = cur.fetchone()[0]

    # Tabla relacional simple. El id es la posicion de la frase en el JSON (0..N-1),
    # asi el mismo id identifica la misma frase en PostgreSQL y en Chroma.
    cur.execute("DROP TABLE IF EXISTS embeddings; DROP TABLE IF EXISTS corpus;")
    cur.execute("CREATE TABLE corpus (id INT PRIMARY KEY, sentence TEXT NOT NULL);")
    conn.commit()

    times = []
    n_statements = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        rows = [(start + j, s) for j, s in enumerate(chunk)]
        # page_size=len(rows): un unico INSERT multi-fila por lote
        # (por defecto psycopg2 parte el lote en trozos de 100 filas)
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
    print("\n--- [P0] INSERCION DEL TEXTO (por lote, incluye commit) ---")
    print(f"Filas insertadas: {n_rows} | lotes: {len(times)} de {BATCH_SIZE} "
          f"| sentencias INSERT: {n_statements}")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {sum(times):.4f} s")

    save_results("postgres_P0.json", {
        "postgres_version": pg_version, "python": platform.python_version(),
        "machine": platform.platform(), "batch_size": BATCH_SIZE,
        "rows": n_rows, "insert_statements": n_statements,
        "stats_seconds": s, "batch_times_seconds": times,
    })


if __name__ == "__main__":
    main()