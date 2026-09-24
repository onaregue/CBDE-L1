import json
import time
import psycopg2
from sentence_transformers import SentenceTransformer
import numpy as np

DB_CONFIG = {
    "dbname": "cbde_l1",
    "user": "postgres",
    "password": "postgrespassword",  # Canvia-ho per la teva contrasenya
    "host": "localhost",
    "port": 5432
}

with open("../query_sentences.json", "r", encoding="utf-8") as f:
    query_sentences = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")
query_embeddings = model.encode(query_sentences)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

cur.execute("""
    -- 1. Distància Euclidiana (L2)
    CREATE OR REPLACE FUNCTION l2_dist(a FLOAT8[], b FLOAT8[]) RETURNS FLOAT8 AS $$
        SELECT SQRT(SUM((x - y)^2)) FROM unnest(a, b) AS t(x, y);
    $$ LANGUAGE sql IMMUTABLE;

    -- 2. Distància Cosinus (1 - similitud del cosinus)
    CREATE OR REPLACE FUNCTION cosine_dist(a FLOAT8[], b FLOAT8[]) RETURNS FLOAT8 AS $$
        SELECT 1.0 - (SUM(x * y) / (SQRT(SUM(x^2)) * SQRT(SUM(y^2)))) 
        FROM unnest(a, b) AS t(x, y);
    $$ LANGUAGE sql IMMUTABLE;
""")
conn.commit()

def test_queries(func_name):
    query_times = []
    for idx, emb in enumerate(query_embeddings):
        emb_list = emb.tolist()
        t0 = time.perf_counter()
        query = f"""
            SELECT id, sentence, {func_name}(embedding, %s) AS dist
            FROM corpus
            WHERE {func_name}(embedding, %s) > 0.0001
            ORDER BY dist ASC
            LIMIT 2;
        """
        cur.execute(query, (emb_list, emb_list))
        _ = cur.fetchall()
        t1 = time.perf_counter()
        query_times.append(t1 - t0)
    return query_times

times_l2 = test_queries("l2_dist")
times_cos = test_queries("cosine_dist")

cur.close()
conn.close()

def print_stats(times, label):
    print(f"\n--- MÈTRIQUES QUERY TOP-2: {label} ---")
    print(f"Mínim: {np.min(times):.4f} s")
    print(f"Màxim: {np.max(times):.4f} s")
    print(f"Mitjana: {np.mean(times):.4f} s")
    print(f"Desviació Estàndard: {np.std(times):.4f} s")

print_stats(times_l2, "Distància Euclidiana (L2)")
print_stats(times_cos, "Distància Cosinus")