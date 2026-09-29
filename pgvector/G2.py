"""
G2 - Compute top-2 most similar sentences for 10 queries in Pgvector using HNSW index (<-> and <=>).
"""
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"
PG_RESULTS_PATH = RESULTS_DIR / "postgres_P2.json"
DB_NAME = os.getenv("PGDATABASE")
HOST = os.getenv("PGHOST")
PORT = os.getenv("PGPORT")
USER = os.getenv("PGUSER")
PASSWORD = os.getenv("PGPASSWORD")

TOP_K = 2
EF_SEARCH = 40
METRICS = ["l2", "cosine"]
OPS = {"l2": "<->", "cosine": "<=>"}

SQL_TPL = """
    SELECT t.id, c.sentence, t.dist
    FROM (
        SELECT e.id, e.embedding {op} (SELECT embedding FROM embeddings_g WHERE id = %s) AS dist
        FROM embeddings_g e
        WHERE e.id <> %s
        ORDER BY e.embedding {op} (SELECT embedding FROM embeddings_g WHERE id = %s)
        LIMIT {k}
    ) t
    JOIN corpus_g c ON c.id = t.id
    ORDER BY t.dist;
"""

def stats(times):
    a = np.asarray(times)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std())
    }

def main():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    conn = psycopg2.connect(
        dbname=DB_NAME, host=HOST, port=PORT, user=USER, password=PASSWORD
    )
    cur = conn.cursor()
    register_vector(conn)

    cur.execute(f"SET enable_seqscan = off; SET hnsw.ef_search = {EF_SEARCH};")

    results, all_stats = {}, {}
    for metric, op in OPS.items():
        sql = SQL_TPL.format(op=op, k=TOP_K)
        # Warm-up run
        cur.execute(sql, (queries[0]["id"], queries[0]["id"], queries[0]["id"]))
        cur.fetchall()

        times, per_query = [], []
        for q in queries:
            t0 = time.perf_counter()
            cur.execute(sql, (q["id"], q["id"], q["id"]))
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
        print(f"\n--- [G2] TOP-{TOP_K} QUERY TIME (Pgvector HNSW, {len(queries)} queries) ---")
        print(f"Min:  {s['min'] * 1000:.3f} ms")
        print(f"Max:  {s['max'] * 1000:.3f} ms")
        print(f"Mean: {s['mean'] * 1000:.3f} ms")
        print(f"Standard deviation: {s['std'] * 1000:.3f} ms")
        print(f"Total time: {sum(r['time_seconds'] for r in results[metric]) * 1000:.3f} ms")

    same = sum(
        [n["id"] for n in ra["neighbors"]] == [n["id"] for n in rb["neighbors"]]
        for ra, rb in zip(results["l2"], results["cosine"])
    )
    print(f"\nQueries with identical top-{TOP_K} ranking between L2 and Cosine: {same}/{len(queries)}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "pgvector_G2.json", "w", encoding="utf-8") as f:
        json.dump({
            "python": platform.python_version(),
            "machine": platform.platform(),
            "metrics": METRICS,
            "ef_search": EF_SEARCH,
            "stats_seconds": all_stats,
            "results": results
        }, f, indent=2)


if __name__ == "__main__":
    main()