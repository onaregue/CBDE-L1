"""
C1 - Genera los embeddings de cada frase y los guarda en Chroma.

A diferencia de C0, aqui los embeddings se calculan FUERA de Chroma (SentenceTransformer,
igual que en P1) y se pasan a add(embeddings=...). Asi la generacion y el almacenamiento
se miden por separado, como en PostgreSQL.

Una coleccion de Chroma tiene UNA sola metrica (se fija al crearla), asi que se crean dos
colecciones (corpus_l2 y corpus_cosine). Los embeddings se generan una vez por lote y se
guardan en ambas; el almacenamiento se mide por coleccion.

Ojo: en Chroma no existe "guardar solo el embedding": cada add() guarda a la vez el
vector (indice HNSW), el documento y el id. Por eso el tiempo de almacenamiento incluye
tambien el texto (en PostgreSQL eran dos INSERT en dos tablas).

Uso (despues de prepare_data.py):
    python Chroma/C1.py

Opciones (variables de entorno, todas opcionales):
    BATCH_SIZE (500), DEVICE (cpu), METRICS ("l2,cosine"), CHROMA_PATH (<repo>/chroma_db)
"""
import json
import os
import platform
import sys
import time
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHROMA_PATH = os.getenv("CHROMA_PATH", str(ROOT / "chroma_db"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
DEVICE = os.getenv("DEVICE", "cpu")
METRICS = [m.strip() for m in os.getenv("METRICS", "l2,cosine").split(",")]
MODEL_NAME = "all-MiniLM-L6-v2"
VALID_METRICS = ("l2", "cosine", "ip")  # las unicas que soporta Chroma


def stats(times):
    a = np.asarray(times)
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def save_results(name, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def print_stats(title, s, total):
    print(f"\n--- {title} ---")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {total:.4f} s")


def make_collection(client, name, metric):
    # embedding_function=None: los vectores los ponemos nosotros; asi Chroma no carga
    # su funcion de embedding por defecto (ONNX), que daria vectores ligeramente distintos.
    try:
        return client.create_collection(
            name, configuration={"hnsw": {"space": metric}}, embedding_function=None)
    except (TypeError, ValueError):  # versiones antiguas de Chroma
        return client.create_collection(
            name, metadata={"hnsw:space": metric}, embedding_function=None)


def main():
    for m in METRICS:
        if m not in VALID_METRICS:
            sys.exit(f"Metrica no soportada por Chroma: {m}. Opciones: {', '.join(VALID_METRICS)}")

    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    model.encode(sentences[:8], show_progress_bar=False)  # calentamiento (no se mide)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    cols = {}
    for m in METRICS:
        try:
            client.delete_collection(f"corpus_{m}")
        except Exception:
            pass
        cols[m] = make_collection(client, f"corpus_{m}", m)

    gen_times, norms = [], []
    store_times = {m: [] for m in METRICS}
    n_calls = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        texts = sentences[start:start + BATCH_SIZE]

        # 1) Generacion (fuera de Chroma), medida por separado
        t0 = time.perf_counter()
        embs = model.encode(texts, batch_size=64, show_progress_bar=False)
        gen_times.append(time.perf_counter() - t0)
        norms.append(np.linalg.norm(embs, axis=1))

        # 2) Almacenamiento: vector + documento + id en una sola llamada. No hay que
        #    convertir a literal SQL: Chroma acepta los arrays de numpy tal cual.
        for m in METRICS:
            t0 = time.perf_counter()
            cols[m].add(ids=[str(start + j) for j in range(len(texts))],
                        embeddings=list(embs), documents=texts)
            store_times[m].append(time.perf_counter() - t0)
            n_calls += 1

    n_rows = {m: cols[m].count() for m in METRICS}
    size = dir_size_mb(CHROMA_PATH)
    norms = np.concatenate(norms)
    gen_s = stats(gen_times)
    store_s = {m: stats(store_times[m]) for m in METRICS}

    print(f"\nRegistros: {n_rows} (dim {embs.shape[1]}) | llamadas a add(): {n_calls} "
          f"| tamano en disco (todas las colecciones): {size:.1f} MB")
    print(f"Norma L2 de los vectores: min {norms.min():.4f}, max {norms.max():.4f}")
    print_stats(f"[C1] GENERACION DE EMBEDDINGS (por lote de {BATCH_SIZE}, device={DEVICE})",
                gen_s, sum(gen_times))
    for m in METRICS:
        print_stats(f"[C1] ALMACENAMIENTO vector+texto (coleccion {m}, por lote)",
                    store_s[m], sum(store_times[m]))

    save_results("chroma_C1.json", {
        "chromadb": chromadb.__version__, "python": platform.python_version(),
        "machine": platform.platform(), "model": MODEL_NAME, "device": DEVICE,
        "metrics": METRICS, "batch_size": BATCH_SIZE, "rows": n_rows,
        "add_calls": n_calls, "size_mb": size,
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "generation_stats_seconds": gen_s, "storage_stats_seconds": store_s,
        "generation_times_seconds": gen_times, "storage_times_seconds": store_times,
    })


if __name__ == "__main__":
    main()