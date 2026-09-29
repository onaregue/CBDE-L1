"""
P2 - Compute top-2 most similar sentences for 10 queries in PostgreSQL using L2 and Cosine
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"

DB_NAME = os.getenv("PGDATABASE")
HOST = os.getenv("PGHOST")
PORT = os.getenv("PGPORT")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")

METRICS = ["l2", "cosine"]
TOP_K = 2

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
        return psycopg2.connect(dbname=DB_NAME, host=HOST, port=PORT, user=USER, password=PASSWORD)
    except psycopg2.OperationalError as e:
        sys.exit(f"Error connecting to PostgreSQL: {e}")


def main():

    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    conn = connect()
    cur = conn.cursor()
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
    for metric, fn in [("l2", "l2_dist"), ("cosine", "cosine_dist")]:
        sql = sql_tpl.format(fn=fn, k=TOP_K)
        # Warm-up run
        cur.execute(sql, (queries[0]["id"], queries[0]["id"]))
        cur.fetchall()

        times, per_query = [], []
        for q in queries:
            t0 = time.perf_counter()
            cur.execute(sql, (q["id"], q["id"]))
            rows = cur.fetchall()
            times.append(time.perf_counter() - t0)
            per_query.append({
                "query_id": q["id"],
                "query_text": q["text"],
                "neighbors": [{"id": r[0], "text": r[1], "distance": float(r[2])} for r in rows],
                "time_seconds": times[-1]
            })
        results[metric] = per_query
        all_stats[metric] = stats(times)

    cur.close()
    conn.close()

    for metric in METRICS:
        print(f"\n=== Metric: {metric.upper()} ===")
        for r in results[metric]:
            print(f"[{r['query_id']}] {r['query_text'][:70]}")
            for n in r["neighbors"]:
                print(f"     -> id {n['id']} (d={n['distance']:.4f}): {n['text'][:70]}")
        s = all_stats[metric]
        print(f"\n--- [P2] TOP-{TOP_K} QUERY TIME ({metric.upper()}, {len(queries)} queries) ---")
        print(f"Min:  {s['min']:.6f} s")
        print(f"Max:  {s['max']:.6f} s")
        print(f"Mean: {s['mean']:.6f} s")
        print(f"Standard deviation: {s['std']:.6f} s")
        print(f"Total time: {sum(r['time_seconds'] for r in results[metric]):.4f} s")

    same = sum(
        [n["id"] for n in ra["neighbors"]] == [n["id"] for n in rb["neighbors"]]
        for ra, rb in zip(results["l2"], results["cosine"])
    )
    print(f"\nQueries with identical top-{TOP_K} ranking between L2 and Cosine: {same}/{len(queries)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "postgres_P2.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(),
            "machine": platform.platform(),
            "metrics": METRICS,
            "stats_seconds": all_stats,
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    main()