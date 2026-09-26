"""
check.py - Verifica de forma independiente los resultados de PostgreSQL/P2.py.

Lee los embeddings guardados en PostgreSQL, recalcula las distancias con numpy y comprueba,
para cada metrica y cada consulta de results/postgres_P2.json, que:
  1. la propia frase de consulta NO aparece entre sus vecinos,
  2. los 2 vecinos son los correctos (un empate de distancia dentro de la tolerancia se
     acepta aunque cambie el orden),
  3. las distancias que devolvio el SQL coinciden con las de numpy,
  4. los vecinos vienen ordenados de menor a mayor distancia,
  5. el texto de cada vecino coincide con el de la tabla corpus.

Uso (despues de ejecutar P0, P1 y P2):
    python check.py
    python check.py --reencode   # ademas comprueba que los embeddings guardados en
                                 # PostgreSQL de las consultas son los que da el modelo

Conexion: mismas variables de entorno que los scripts (PGDATABASE, PGHOST, PGUSER, ...).
Sale con codigo 0 si todo es correcto y con 1 si hay algun fallo.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg2

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "postgres_P2.json"
QUERIES_PATH = ROOT / "query_sentences.json"
DB_NAME = os.getenv("PGDATABASE", "cbde_l1")
HOST = os.getenv("PGHOST", "localhost")
DEVICE = os.getenv("DEVICE", "cpu")
MODEL_NAME = "all-MiniLM-L6-v2"
TOL = 1e-6          # tolerancia entre distancia del SQL y la de numpy
REENCODE_TOL = 1e-4  # tolerancia entre embedding guardado y embedding recalculado


def distances(metric, E, v):
    if metric == "l2":
        return np.linalg.norm(E - v, axis=1)
    if metric == "cosine":
        # suma explicita en lugar de "E @ v": evita la libreria BLAS (en macOS,
        # Accelerate emite avisos de coma flotante espurios en el producto matricial)
        dots = (E * v).sum(axis=1)
        return 1.0 - dots / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))
    if metric == "l1":
        return np.abs(E - v).sum(axis=1)
    sys.exit(f"Metrica desconocida en los resultados: {metric}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reencode", action="store_true",
                    help="recalcula con el modelo el embedding de las consultas y lo compara")
    args = ap.parse_args()

    for p in (RESULTS_PATH, QUERIES_PATH):
        if not p.exists():
            sys.exit(f"Falta {p}: ejecuta antes prepare_data.py, P0, P1 y P2.")
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)["results"]
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)

    try:
        conn = psycopg2.connect(dbname=DB_NAME, host=HOST)
    except psycopg2.OperationalError as e:
        sys.exit(f"No se pudo conectar a PostgreSQL: {e}")
    cur = conn.cursor()
    cur.execute("""SELECT c.id, c.sentence, e.embedding
                   FROM corpus c JOIN embeddings e USING (id) ORDER BY c.id;""")
    rows = cur.fetchall()
    conn.close()

    ids = [r[0] for r in rows]
    pos = {i: k for k, i in enumerate(ids)}          # id -> fila en la matriz
    sentences = [r[1] for r in rows]
    E = np.array([r[2] for r in rows], dtype=np.float64)  # valores float32 guardados
    norms = np.linalg.norm(E, axis=1)
    print(f"Corpus: {len(ids)} frases, dimension {E.shape[1]}, "
          f"norma L2 min {norms.min():.4f} / max {norms.max():.4f}")

    failures = []

    def fail(msg):
        failures.append(msg)
        print(f"   FALLO: {msg}")

    for metric, per_query in results.items():
        ok_queries = 0
        if len(per_query) != len(queries):
            fail(f"[{metric}] hay {len(per_query)} resultados y {len(queries)} consultas")
        for q, r in zip(queries, per_query):
            n_fail_before = len(failures)
            qid = q["id"]
            if r["query_id"] != qid or r["query_text"] != q["text"]:
                fail(f"[{metric}] el resultado {r['query_id']} no corresponde a la "
                     f"consulta {qid} de query_sentences.json")
                continue
            if qid not in pos:
                fail(f"[{metric}] la consulta {qid} no esta en el corpus")
                continue

            d = distances(metric, E, E[pos[qid]])
            if not np.isfinite(d).all():
                fail(f"[{metric}] {qid}: hay distancias no finitas (NaN/inf) en numpy")
                continue
            d[pos[qid]] = np.inf                       # excluir la propia frase
            expected = np.argsort(d)[:len(r["neighbors"])]
            got = [n["id"] for n in r["neighbors"]]
            got_pos = [pos[i] for i in got if i in pos]

            if qid in got:                                                     # (1)
                fail(f"[{metric}] {qid}: la propia frase aparece entre sus vecinos")
            if len(got_pos) != len(got):
                fail(f"[{metric}] {qid}: algun vecino no existe en el corpus")
                continue
            if got != [ids[k] for k in expected]:                              # (2)
                if not np.allclose(d[got_pos], d[expected], atol=TOL, rtol=0):
                    fail(f"[{metric}] {qid}: vecinos {got} != esperados "
                         f"{[ids[k] for k in expected]}")
            reported = np.array([n["distance"] for n in r["neighbors"]])
            if not np.allclose(reported, d[got_pos], atol=TOL, rtol=0):        # (3)
                fail(f"[{metric}] {qid}: distancias SQL {reported.round(6).tolist()} "
                     f"!= numpy {d[got_pos].round(6).tolist()}")
            if list(reported) != sorted(reported):                             # (4)
                fail(f"[{metric}] {qid}: vecinos no ordenados por distancia")
            for n in r["neighbors"]:                                           # (5)
                if n["text"] != sentences[pos[n["id"]]]:
                    fail(f"[{metric}] {qid}: el texto del vecino {n['id']} no coincide")
            if len(failures) == n_fail_before:
                ok_queries += 1
        print(f"{metric:>7}: {ok_queries}/{len(queries)} consultas correctas")

    if args.reencode:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        fresh = model.encode([q["text"] for q in queries], show_progress_bar=False)
        diffs = [np.abs(fresh[k] - E[pos[q["id"]]]).max() for k, q in enumerate(queries)]
        print(f"reencode: diferencia maxima embedding guardado vs modelo = {max(diffs):.2e}")
        if max(diffs) > REENCODE_TOL:
            fail("el embedding guardado en PostgreSQL no coincide con el del modelo")

    print("\nRESULTADO:", "TODO CORRECTO" if not failures else f"{len(failures)} FALLO(S)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()