"""
P2 - Para 10 frases del corpus (query_sentences.json, generado por prepare_data.py),
calcula las 2 frases mas similares ENTRE TODAS LAS DEMAS con dos metricas de distancia.
La distancia se calcula en SQL (funciones propias sobre arrays): sin pgvector.

Se usa el embedding ya guardado por P1 para la frase de consulta y se excluye a la
propia frase por id.

Uso (despues de P0 y P1):
    python PostgreSQL/P2.py

Conexion: mismas variables de entorno que P0 (PGDATABASE, PGHOST, PGPORT, PGUSER, PGPASSWORD).
Otras opciones:
    METRICS  metricas separadas por comas (por defecto "l2,cosine"; tambien "l1").
             Chroma solo soporta l2, cosine e ip, asi que para comparar usad l2 y cosine.
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
METRICS = [m.strip() for m in os.getenv("METRICS", "l2,cosine").split(",")]
TOP_K = 2

FUNC_NAMES = {"l2": "l2_dist", "cosine": "cosine_dist", "l1": "l1_dist"}

# Acumulamos en float8 (sum(real) devolveria real y perderia precision).
FUNCTIONS_SQL = """
CREATE OR REPLACE FUNCTION l2_dist(a REAL[], b REAL[]) RETURNS FLOAT8 AS $$
    SELECT sqrt(sum((x::float8 - y::float8) * (x::float8 - y::float8)))
    FROM unnest(a, b) AS t(x, y)
$$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION cosine_dist(a REAL[], b REAL[]) RETURNS FLOAT8 AS $$
    SELECT 1.0 - sum(x::float8 * y::float8)
                 / (sqrt(sum(x::float8 * x::float8)) * sqrt(sum(y::float8 * y::float8)))
    FROM unnest(a, b) AS t(x, y)
$$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;

CREATE OR REPLACE FUNCTION l1_dist(a REAL[], b REAL[]) RETURNS FLOAT8 AS $$
    SELECT sum(abs(x::float8 - y::float8))
    FROM unnest(a, b) AS t(x, y)
$$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;
"""


def connect():
    try:
        return psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(
            f"No se pudo conectar a PostgreSQL: {e}\n"
            "Comprueba que el servidor esta arrancado, que has ejecutado P0 y P1 y, "
            "si hace falta, define PGUSER / PGPASSWORD / PGHOST / PGPORT."
        )


def stats(times):
    a = np.asarray(times)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def load_queries():
    if not QUERIES_PATH.exists():
        sys.exit(f"Falta {QUERIES_PATH.name}: ejecuta prepare_data.py (o haz git pull).")
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    if (not queries or not isinstance(queries[0], dict)
            or not isinstance(queries[0].get("id"), int) or "text" not in queries[0]):
        sys.exit(f'{QUERIES_PATH.name} debe ser una lista de {{"id": int, "text": str}}: '
                 "regeneralo con prepare_data.py.")
    return queries


def main():
    for m in METRICS:
        if m not in FUNC_NAMES:
            sys.exit(f"Metrica desconocida: {m}. Opciones: {', '.join(FUNC_NAMES)}")
    queries = load_queries()

    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM embeddings;")
    if cur.fetchone()[0] == 0:
        sys.exit("La tabla embeddings esta vacia: ejecuta primero P0 y P1.")
    cur.execute(FUNCTIONS_SQL)
    conn.commit()

    # Se ordena y limita en la subconsulta y solo despues se une con corpus para
    # recuperar el texto de los 2 vecinos. "e.id <> %s" excluye la propia frase.
    sql_tpl = """
        SELECT t.id, c.sentence, t.dist
        FROM (
            SELECT e.id, {fn}(e.embedding, q.embedding) AS dist
            FROM embeddings e,
                 (SELECT embedding FROM embeddings WHERE id = %s) q
            WHERE e.id <> %s
            ORDER BY dist
            LIMIT {k}
        ) t
        JOIN corpus c ON c.id = t.id
        ORDER BY t.dist;
    """

    results, all_stats = {}, {}
    for metric in METRICS:
        sql = sql_tpl.format(fn=FUNC_NAMES[metric], k=TOP_K)
        qid0 = queries[0]["id"]
        cur.execute(sql, (qid0, qid0))  # calentamiento (cache fria): no se mide
        cur.fetchall()

        times, per_query = [], []
        for q in queries:
            t0 = time.perf_counter()
            cur.execute(sql, (q["id"], q["id"]))
            rows = cur.fetchall()
            times.append(time.perf_counter() - t0)
            per_query.append({
                "query_id": q["id"], "query_text": q["text"],
                "neighbors": [{"id": r[0], "text": r[1], "distance": r[2]} for r in rows],
                "time_seconds": times[-1],
            })
        results[metric] = per_query
        all_stats[metric] = stats(times)

    cur.close()
    conn.close()

    for metric in METRICS:
        print(f"\n=== Metrica: {metric} ===")
        for r in results[metric]:
            print(f"[{r['query_id']}] {r['query_text'][:70]}")
            for n in r["neighbors"]:
                print(f"     -> id {n['id']} (d={n['distance']:.4f}): {n['text'][:70]}")
        s = all_stats[metric]
        print(f"--- [P2] TIEMPOS TOP-{TOP_K} ({metric}, {len(queries)} consultas) ---")
        print(f"Min:  {s['min']:.4f} s")
        print(f"Max:  {s['max']:.4f} s")
        print(f"Media: {s['mean']:.4f} s")
        print(f"Desviacion estandar: {s['std']:.4f} s")

    if len(METRICS) >= 2:
        a, b = METRICS[0], METRICS[1]
        same = sum(
            [n["id"] for n in ra["neighbors"]] == [n["id"] for n in rb["neighbors"]]
            for ra, rb in zip(results[a], results[b])
        )
        print(f"\nConsultas con el mismo top-{TOP_K} en {a} y {b}: {same}/{len(queries)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "postgres_P2.json", "w", encoding="utf-8") as f:
        json.dump({"python": platform.python_version(), "machine": platform.platform(),
                   "metrics": METRICS, "stats_seconds": all_stats, "results": results},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()