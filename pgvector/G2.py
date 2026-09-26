"""
G2 - Top-2 vecinos mas similares (entre todas las demas frases) de las mismas 10 consultas
de query_sentences.json, con los operadores de pgvector: <-> (L2) y <=> (coseno).

Se ejecuta en dos modos, para separar el efecto del tipo de dato del efecto del indice:
  exact : indices desactivados -> barrido secuencial + ordenacion (busqueda EXACTA, la
          misma semantica que P2, pero con el operador nativo en C en vez de SQL sobre arrays)
  hnsw  : usa el indice HNSW de G1 (busqueda APROXIMADA, como Chroma)
Se comprueba con EXPLAIN que cada modo usa (o no) el indice de verdad.

Ademas, sin medir tiempos, se calcula el recall@2 del modo hnsw frente al exacto sobre
N_SAMPLE frases al azar, y se compara el modo exacto con results/postgres_P2.json.

Uso (despues de G0 y G1):
    python Pgvector/G2.py

Opciones (entorno): METRICS ("l2,cosine"), MODES ("exact,hnsw"), EF_SEARCH (40, valor por
defecto de pgvector), N_SAMPLE (200).
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

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"
PG_RESULTS_PATH = RESULTS_DIR / "postgres_P2.json"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
METRICS = [m.strip() for m in os.getenv("METRICS", "l2,cosine").split(",")]
MODES = [m.strip() for m in os.getenv("MODES", "exact,hnsw").split(",")]
EF_SEARCH = int(os.getenv("EF_SEARCH", "40"))
N_SAMPLE = int(os.getenv("N_SAMPLE", "200"))
TOP_K = 2
OPS = {"l2": "<->", "cosine": "<=>"}

# ORDER BY repite la expresion del operador (sin alias) para que el planificador pueda
# usar el indice. "e.id <> %s" excluye la propia frase. El texto se une al final.
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
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def load_queries():
    if not QUERIES_PATH.exists():
        sys.exit(f"Falta {QUERIES_PATH.name}: ejecuta prepare_data.py.")
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)


def set_mode(cur, mode):
    cur.execute("RESET enable_indexscan; RESET enable_seqscan;")
    if mode == "exact":
        cur.execute("SET enable_indexscan = off;")
    else:
        cur.execute(f"SET enable_seqscan = off; SET hnsw.ef_search = {EF_SEARCH};")


def uses_index(cur, sql, qid):
    cur.execute("EXPLAIN " + sql, (qid, qid, qid))
    return "hnsw" in "\n".join(r[0] for r in cur.fetchall()).lower()


def run(cur, sql, qid):
    cur.execute(sql, (qid, qid, qid))
    return cur.fetchall()


def main():
    for m in METRICS:
        if m not in OPS:
            sys.exit(f"Metrica no soportada: {m}. Opciones: {', '.join(OPS)}")
    queries = load_queries()

    try:
        conn = psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(f"No se pudo conectar a PostgreSQL: {e}")
    cur = conn.cursor()
    try:
        register_vector(conn)
        cur.execute("SELECT count(*) FROM embeddings_g;")
    except psycopg2.Error:
        conn.rollback()
        sys.exit("Falta la extension o la tabla embeddings_g: ejecuta primero G0 y G1.")
    n_rows = cur.fetchone()[0]

    results = {mode: {} for mode in MODES}
    all_stats = {mode: {} for mode in MODES}
    plan_ok = {mode: {} for mode in MODES}
    for mode in MODES:
        set_mode(cur, mode)
        for metric in METRICS:
            sql = SQL_TPL.format(op=OPS[metric], k=TOP_K)
            used = uses_index(cur, sql, queries[0]["id"])
            plan_ok[mode][metric] = used == (mode == "hnsw")
            if not plan_ok[mode][metric]:
                print(f"AVISO: en modo {mode}/{metric} el plan "
                      f"{'usa' if used else 'NO usa'} el indice HNSW (no es lo esperado).")
            run(cur, sql, queries[0]["id"])  # calentamiento: no se mide

            times, per_query = [], []
            for q in queries:
                t0 = time.perf_counter()
                rows = run(cur, sql, q["id"])
                times.append(time.perf_counter() - t0)
                per_query.append({
                    "query_id": q["id"], "query_text": q["text"],
                    "neighbors": [{"id": r[0], "text": r[1], "distance": float(r[2])}
                                  for r in rows],
                    "time_seconds": times[-1]})
            results[mode][metric] = per_query
            all_stats[mode][metric] = stats(times)

    for mode in MODES:
        for metric in METRICS:
            print(f"\n=== Modo: {mode} | Metrica: {metric} ===")
            for r in results[mode][metric]:
                print(f"[{r['query_id']}] {r['query_text'][:70]}")
                for n in r["neighbors"]:
                    print(f"     -> id {n['id']} (d={n['distance']:.4f}): {n['text'][:70]}")
            s = all_stats[mode][metric]
            print(f"--- [G2] TIEMPOS TOP-{TOP_K} ({mode}, {metric}, {len(queries)} consultas) ---")
            print(f"Min:  {s['min']:.6f} s")
            print(f"Max:  {s['max']:.6f} s")
            print(f"Media: {s['mean']:.6f} s")
            print(f"Desviacion estandar: {s['std']:.6f} s")

    # --- Comparaciones (no miden tiempo) ---
    extra = {"recall": {}, "vs_postgres_exact": {}, "hnsw_vs_exact_10": {}}
    if "exact" in MODES and "hnsw" in MODES:
        print("\n=== HNSW vs exacto ===")
        rng = np.random.default_rng(42)
        sample = [int(i) for i in rng.choice(n_rows, size=min(N_SAMPLE, n_rows), replace=False)]
        for metric in METRICS:
            same10 = sum(
                {n["id"] for n in a["neighbors"]} == {n["id"] for n in b["neighbors"]}
                for a, b in zip(results["exact"][metric], results["hnsw"][metric]))
            sql = SQL_TPL.format(op=OPS[metric], k=TOP_K)
            got = {}
            for mode in ("exact", "hnsw"):
                set_mode(cur, mode)
                got[mode] = {qid: {r[0] for r in run(cur, sql, qid)} for qid in sample}
            hits = sum(len(got["exact"][q] & got["hnsw"][q]) for q in sample)
            exact_sets = sum(got["exact"][q] == got["hnsw"][q] for q in sample)
            recall = hits / (TOP_K * len(sample))
            extra["recall"][metric] = recall
            extra["hnsw_vs_exact_10"][metric] = same10
            print(f"{metric:>7}: 10 consultas con el mismo conjunto {same10}/{len(queries)} | "
                  f"recall@{TOP_K} sobre {len(sample)} frases = {recall:.4f} "
                  f"({exact_sets}/{len(sample)} conjuntos exactos)")

    if "exact" in MODES and PG_RESULTS_PATH.exists():
        with open(PG_RESULTS_PATH, encoding="utf-8") as f:
            pg = json.load(f)["results"]
        print("\n=== pgvector (exacto) vs PostgreSQL P2 (UDF en SQL) ===")
        for metric in METRICS:
            if metric not in pg:
                continue
            same = 0
            max_diff = 0.0
            for rg, rp in zip(results["exact"][metric], pg[metric]):
                ig = [n["id"] for n in rg["neighbors"]]
                ip = [n["id"] for n in rp["neighbors"]]
                if ig == ip:
                    same += 1
                    max_diff = max(max_diff, max(abs(x["distance"] - y["distance"])
                                                 for x, y in zip(rg["neighbors"], rp["neighbors"])))
            extra["vs_postgres_exact"][metric] = {"same_order": same, "max_distance_diff": max_diff}
            print(f"{metric:>7}: mismos vecinos y orden {same}/{len(queries)}, "
                  f"dif. maxima de distancia {max_diff:.2e}")

    cur.close()
    conn.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "pgvector_G2.json", "w", encoding="utf-8") as f:
        json.dump({"python": platform.python_version(), "machine": platform.platform(),
                   "metrics": METRICS, "modes": MODES, "ef_search": EF_SEARCH,
                   "plan_as_expected": plan_ok, "stats_seconds": all_stats,
                   "comparisons": extra, "results": results},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()