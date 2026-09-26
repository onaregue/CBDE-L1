"""
C2 - Para las mismas 10 frases de query_sentences.json, calcula las 2 frases mas
similares ENTRE TODAS LAS DEMAS con dos metricas (l2 y cosine), usando Chroma.

Igual que en P2 se usa el embedding YA guardado (por C1) para la frase de consulta, no se
vuelve a generar. En P2 eso eran una subconsulta y un JOIN dentro de una sola sentencia
SQL; aqui son dos llamadas: get() para recuperar el vector y query() para buscar. Ambas
entran en el tiempo medido. La propia frase se descarta por id (se piden k+1 vecinos).

Si existe results/postgres_P2.json, compara los ids de los vecinos con los de PostgreSQL.
OJO: Chroma busca con un indice HNSW (aproximado); PostgreSQL recorre todas las filas
(exacto). En "l2" Chroma devuelve la distancia AL CUADRADO: aqui se guarda tambien su raiz
para poder compararla con la de PostgreSQL.

Uso (despues de prepare_data.py y C1):
    python Chroma/C2.py

Opciones (variables de entorno): CHROMA_PATH (<repo>/chroma_db), METRICS ("l2,cosine").
"""
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
QUERIES_PATH = ROOT / "query_sentences.json"
PG_RESULTS_PATH = RESULTS_DIR / "postgres_P2.json"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_db"))
METRICS = [m.strip() for m in os.getenv("METRICS", "l2,cosine").split(",")]
TOP_K = 2


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
        sys.exit(f'{QUERIES_PATH.name} debe ser una lista de {{"id": int, "text": str}}.')
    return queries


def search(col, qid):
    """Devuelve (vecinos, t_get, t_query). Vecinos = lista de (id:int, texto, distancia)."""
    t0 = time.perf_counter()
    got = col.get(ids=[str(qid)], include=["embeddings"])
    t1 = time.perf_counter()
    res = col.query(query_embeddings=[got["embeddings"][0]], n_results=TOP_K + 1,
                    include=["documents", "distances"])
    t2 = time.perf_counter()
    neighbors = [(int(i), doc, float(d)) for i, doc, d in
                 zip(res["ids"][0], res["documents"][0], res["distances"][0])
                 if i != str(qid)][:TOP_K]
    return neighbors, t1 - t0, t2 - t1


def main():
    for m in METRICS:
        if m not in ("l2", "cosine", "ip"):
            sys.exit(f"Metrica no soportada por Chroma: {m}")
    queries = load_queries()

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

    results, all_stats = {}, {}
    for metric in METRICS:
        try:
            col = client.get_collection(f"corpus_{metric}", embedding_function=None)
        except Exception as e:
            sys.exit(f"No existe la coleccion corpus_{metric} ({e}). Ejecuta antes C1.")

        search(col, queries[0]["id"])  # calentamiento: no se mide

        times, per_query = [], []
        for q in queries:
            neighbors, t_get, t_query = search(col, q["id"])
            times.append(t_get + t_query)
            per_query.append({
                "query_id": q["id"], "query_text": q["text"],
                "neighbors": [{
                    "id": i, "text": txt, "distance_raw": d,
                    # l2 de Chroma = distancia al cuadrado -> raiz para comparar
                    "distance": math.sqrt(max(d, 0.0)) if metric == "l2" else d,
                } for i, txt, d in neighbors],
                "time_seconds": times[-1], "time_get_seconds": t_get,
                "time_query_seconds": t_query,
            })
        results[metric] = per_query
        all_stats[metric] = stats(times)

    for metric in METRICS:
        print(f"\n=== Metrica: {metric} ===")
        for r in results[metric]:
            print(f"[{r['query_id']}] {r['query_text'][:70]}")
            for n in r["neighbors"]:
                print(f"     -> id {n['id']} (d={n['distance']:.4f}): {n['text'][:70]}")
        s = all_stats[metric]
        print(f"--- [C2] TIEMPOS TOP-{TOP_K} ({metric}, {len(queries)} consultas; get + query) ---")
        print(f"Min:  {s['min']:.4f} s")
        print(f"Max:  {s['max']:.4f} s")
        print(f"Media: {s['mean']:.4f} s")
        print(f"Desviacion estandar: {s['std']:.4f} s")

    if len(METRICS) >= 2:
        a, b = METRICS[0], METRICS[1]
        same = sum([n["id"] for n in ra["neighbors"]] == [n["id"] for n in rb["neighbors"]]
                   for ra, rb in zip(results[a], results[b]))
        print(f"\nConsultas con el mismo top-{TOP_K} en {a} y {b}: {same}/{len(queries)}")

    comparison = {}
    if PG_RESULTS_PATH.exists():
        with open(PG_RESULTS_PATH, encoding="utf-8") as f:
            pg = json.load(f)["results"]
        print("\n=== Chroma (HNSW, aproximado) vs PostgreSQL (exacto) ===")
        for metric in METRICS:
            if metric not in pg:
                continue
            same_order = same_set = 0
            max_diff = 0.0
            for rc, rp in zip(results[metric], pg[metric]):
                if rc["query_id"] != rp["query_id"]:
                    continue
                ic = [n["id"] for n in rc["neighbors"]]
                ip = [n["id"] for n in rp["neighbors"]]
                same_order += ic == ip
                same_set += set(ic) == set(ip)
                if ic == ip:
                    max_diff = max(max_diff, max(abs(x["distance"] - y["distance"])
                                                 for x, y in zip(rc["neighbors"], rp["neighbors"])))
            comparison[metric] = {"same_order": same_order, "same_set": same_set,
                                  "max_distance_diff": max_diff}
            print(f"{metric:>7}: mismos vecinos y orden {same_order}/{len(queries)}, "
                  f"mismo conjunto {same_set}/{len(queries)}, "
                  f"dif. maxima de distancia {max_diff:.2e}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "chroma_C2.json", "w", encoding="utf-8") as f:
        json.dump({"chromadb": chromadb.__version__, "python": platform.python_version(),
                   "machine": platform.platform(), "metrics": METRICS,
                   "stats_seconds": all_stats, "vs_postgres": comparison,
                   "results": results}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()