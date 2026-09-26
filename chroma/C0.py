"""
C0 - Carga del texto de BookCorpus en Chroma.

En Chroma un registro es (id, vector, documento): no se puede guardar el texto sin su
vector. Por eso, aqui se pasa SOLO el texto a add() y es la propia coleccion quien
genera el embedding (funcion de embedding = mismo modelo que en PostgreSQL, en CPU).
Consecuencia: en C0 el tiempo de add() mezcla generacion del embedding + insercion del
texto + insercion del vector en el indice HNSW, y NO se pueden separar. C1 hace la
version separable (embeddings calculados fuera y pasados a add()).

Uso (desde cualquier carpeta, despues de prepare_data.py):
    python Chroma/C0.py

Opciones (variables de entorno, todas opcionales):
    BATCH_SIZE (500), DEVICE (cpu), METRIC (l2), CHROMA_PATH_C0 (<repo>/chroma_db_c0)
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
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
CHROMA_PATH = os.getenv("CHROMA_PATH_C0", str(ROOT / "chroma_db_c0"))
COLLECTION = "corpus_c0"
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "500"))
DEVICE = os.getenv("DEVICE", "cpu")
METRIC = os.getenv("METRIC", "l2")
MODEL_NAME = "all-MiniLM-L6-v2"


def stats(times):
    a = np.asarray(times)
    # std poblacional (ddof=0): la misma que en los scripts de PostgreSQL
    return {"min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "std": float(a.std())}


def save_results(name, data):
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dir_size_mb(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6


def make_collection(client, name, metric, ef):
    """La metrica se fija al crear la coleccion (no se puede cambiar despues)."""
    try:
        return client.create_collection(
            name, configuration={"hnsw": {"space": metric}}, embedding_function=ef)
    except (TypeError, ValueError):  # versiones antiguas de Chroma
        return client.create_collection(
            name, metadata={"hnsw:space": metric}, embedding_function=ef)


def main():
    with open(ROOT / "bookcorpus_sentences.json", encoding="utf-8") as f:
        sentences = json.load(f)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME, device=DEVICE)
    ef(sentences[:8])  # calentamiento (no se mide)

    client = chromadb.PersistentClient(
        path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = make_collection(client, COLLECTION, METRIC, ef)

    times = []
    n_calls = 0
    for start in range(0, len(sentences), BATCH_SIZE):
        chunk = sentences[start:start + BATCH_SIZE]
        t0 = time.perf_counter()
        # Los ids de Chroma son str: str(posicion) = mismo id que en PostgreSQL.
        # Sin "embeddings=": add() llama a la funcion de embedding por dentro.
        col.add(ids=[str(start + j) for j in range(len(chunk))], documents=chunk)
        times.append(time.perf_counter() - t0)
        n_calls += 1

    n_rows = col.count()
    size = dir_size_mb(CHROMA_PATH)
    s = stats(times)
    print(f"\nRegistros en Chroma: {n_rows} | llamadas a add(): {n_calls} "
          f"| metrica: {METRIC} | tamano en disco: {size:.1f} MB")
    print(f"\n--- [C0] TEXTO + EMBEDDING IMPLICITO (por lote de {BATCH_SIZE}, device={DEVICE}) ---")
    print("(no separable: add() genera el embedding y guarda texto y vector en una sola llamada)")
    print(f"Min:  {s['min']:.6f} s")
    print(f"Max:  {s['max']:.6f} s")
    print(f"Media: {s['mean']:.6f} s")
    print(f"Desviacion estandar: {s['std']:.6f} s")
    print(f"Tiempo total: {sum(times):.4f} s")

    save_results("chroma_C0.json", {
        "chromadb": chromadb.__version__, "python": platform.python_version(),
        "machine": platform.platform(), "model": MODEL_NAME, "device": DEVICE,
        "metric": METRIC, "batch_size": BATCH_SIZE, "rows": n_rows,
        "add_calls": n_calls, "size_mb": size,
        "stats_seconds": s, "batch_times_seconds": times,
    })


if __name__ == "__main__":
    main()