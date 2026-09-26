"""
check_chroma.py - Verifica de forma independiente los resultados de Chroma (C0, C1, C2).

Lee los vectores y documentos guardados en las colecciones de Chroma, recalcula con numpy
(busqueda exacta) y comprueba:
  1. Integridad: 10.000 registros, ids "0".."9999", documentos identicos a
     bookcorpus_sentences.json y vectores normalizados (norma ~1).
  2. Las colecciones corpus_l2 y corpus_cosine tienen los mismos vectores.
  3. (si existe chroma_db_c0) los vectores que genero Chroma por dentro en C0 coinciden
     con los de C1 (calculados fuera).
  4. Para las 10 consultas de results/chroma_C2.json y cada metrica: la propia frase no
     aparece, los 2 vecinos son los exactos (un empate se acepta), las distancias
     coinciden con numpy, estan ordenadas y el texto es el del corpus.
     En l2 tambien se comprueba que distance_raw = distance**2 (Chroma da l2 al cuadrado).
  5. Recall@2 del indice HNSW frente a la busqueda exacta, sobre N_SAMPLE frases al azar
     (las 10 consultas son una muestra pequena para medir un indice aproximado).

Uso (despues de prepare_data.py, C0, C1 y C2):
    python check_chroma.py
Opciones (entorno): CHROMA_PATH, CHROMA_PATH_C0, N_SAMPLE (200).
Sale con codigo 0 si todo es correcto y con 1 si hay algun fallo.
"""
import json
import os
import sys
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results" / "chroma_C2.json"
SENTENCES_PATH = ROOT / "bookcorpus_sentences.json"
QUERIES_PATH = ROOT / "query_sentences.json"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_db"))
CHROMA_PATH_C0 = os.getenv("CHROMA_PATH_C0", str(ROOT / "chroma_db_c0"))
N_SAMPLE = int(os.getenv("N_SAMPLE", "200"))
TOL = 1e-5  # Chroma calcula en float32
TOP_K = 2

failures = []


def fail(msg):
    failures.append(msg)
    print(f"   FALLO: {msg}")


def distances(metric, E, v):
    if metric == "l2":
        return np.linalg.norm(E - v, axis=1)
    if metric == "cosine":
        dots = (E * v).sum(axis=1)
        return 1.0 - dots / (np.linalg.norm(E, axis=1) * np.linalg.norm(v))
    sys.exit(f"Metrica no soportada: {metric}")


def load_collection(client, name, n):
    col = client.get_collection(name, embedding_function=None)
    got = col.get(include=["embeddings", "documents"], limit=n + 1)
    order = np.argsort([int(i) for i in got["ids"]])
    ids = [int(got["ids"][k]) for k in order]
    docs = [got["documents"][k] for k in order]
    E = np.array([got["embeddings"][k] for k in order], dtype=np.float64)
    return col, ids, docs, E


def main():
    for p in (RESULTS_PATH, SENTENCES_PATH, QUERIES_PATH):
        if not p.exists():
            sys.exit(f"Falta {p}: ejecuta antes prepare_data.py, C1 y C2.")
    with open(SENTENCES_PATH, encoding="utf-8") as f:
        sentences = json.load(f)
    with open(QUERIES_PATH, encoding="utf-8") as f:
        queries = json.load(f)
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)["results"]
    n = len(sentences)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))

    data = {}
    for metric in results:
        col, ids, docs, E = load_collection(client, f"corpus_{metric}", n)
        data[metric] = (col, E)
        # (1) integridad
        if col.count() != n or ids != list(range(n)):
            fail(f"[{metric}] la coleccion no tiene los ids 0..{n - 1}")
        if docs != sentences:
            fail(f"[{metric}] los documentos no coinciden con bookcorpus_sentences.json")
        norms = np.linalg.norm(E, axis=1)
        print(f"{metric:>7}: {col.count()} registros, dim {E.shape[1]}, "
              f"norma min {norms.min():.4f} / max {norms.max():.4f}")
        if abs(norms.min() - 1) > 1e-3 or abs(norms.max() - 1) > 1e-3:
            fail(f"[{metric}] los vectores no estan normalizados")

    # (2) mismas colecciones -> mismos vectores
    ms = list(data)
    for m in ms[1:]:
        d = np.abs(data[ms[0]][1] - data[m][1]).max()
        print(f"vectores {ms[0]} vs {m}: dif. maxima {d:.2e}")
        if d > 1e-6:
            fail(f"los vectores de {ms[0]} y {m} difieren")

    # (3) C0 (embedding implicito) vs C1 (embedding externo)
    if Path(CHROMA_PATH_C0).exists():
        c0 = chromadb.PersistentClient(
            path=CHROMA_PATH_C0, settings=Settings(anonymized_telemetry=False))
        _, ids0, docs0, E0 = load_collection(c0, "corpus_c0", n)
        d = np.abs(E0 - data[ms[0]][1]).max()
        print(f"vectores C0 (implicitos) vs C1 (externos): dif. maxima {d:.2e}")
        if ids0 != list(range(n)) or docs0 != sentences or d > 1e-4:
            fail("C0 y C1 no guardan lo mismo")

    # (4) las 10 consultas de C2 contra numpy
    for metric, per_query in results.items():
        E = data[metric][1]
        ok = 0
        for q, r in zip(queries, per_query):
            before = len(failures)
            qid = q["id"]
            if r["query_id"] != qid or r["query_text"] != q["text"]:
                fail(f"[{metric}] resultado {r['query_id']} != consulta {qid}")
                continue
            d = distances(metric, E, E[qid])
            d[qid] = np.inf
            expected = np.argsort(d)[:TOP_K]
            got = [nb["id"] for nb in r["neighbors"]]
            if qid in got:
                fail(f"[{metric}] {qid}: la propia frase aparece entre los vecinos")
            if len(got) != TOP_K:
                fail(f"[{metric}] {qid}: {len(got)} vecinos en vez de {TOP_K}")
                continue
            if got != list(expected) and not np.allclose(d[got], d[expected], atol=TOL, rtol=0):
                fail(f"[{metric}] {qid}: vecinos {got} != exactos {expected.tolist()}")
            rep = np.array([nb["distance"] for nb in r["neighbors"]])
            if not np.allclose(rep, d[got], atol=TOL, rtol=0):
                fail(f"[{metric}] {qid}: distancias {rep.round(6).tolist()} "
                     f"!= numpy {d[got].round(6).tolist()}")
            if list(rep) != sorted(rep):
                fail(f"[{metric}] {qid}: vecinos sin ordenar")
            if metric == "l2":
                raw = np.array([nb["distance_raw"] for nb in r["neighbors"]])
                if not np.allclose(raw, rep ** 2, atol=TOL, rtol=0):
                    fail(f"[{metric}] {qid}: distance_raw no es distance^2")
            for nb in r["neighbors"]:
                if nb["text"] != sentences[nb["id"]]:
                    fail(f"[{metric}] {qid}: texto del vecino {nb['id']} incorrecto")
            ok += len(failures) == before
        print(f"{metric:>7}: {ok}/{len(queries)} consultas correctas frente a numpy")

    # (5) recall@2 de HNSW sobre una muestra mayor
    rng = np.random.default_rng(42)
    sample = rng.choice(n, size=min(N_SAMPLE, n), replace=False)
    for metric, (col, E) in data.items():
        hits = exact_sets = 0
        for qid in sample:
            d = distances(metric, E, E[qid])
            d[qid] = np.inf
            exact = set(np.argsort(d)[:TOP_K].tolist())
            res = col.query(query_embeddings=[E[qid].tolist()], n_results=TOP_K + 1)
            got = [int(i) for i in res["ids"][0] if int(i) != qid][:TOP_K]
            hits += len(exact & set(got))
            exact_sets += exact == set(got)
        print(f"{metric:>7}: recall@{TOP_K} HNSW = {hits / (TOP_K * len(sample)):.4f} "
              f"({exact_sets}/{len(sample)} consultas con el conjunto exacto)")

    print("\nRESULTADO:", "TODO CORRECTO" if not failures else f"{len(failures)} FALLO(S)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()