import json
import time
import psycopg2
from psycopg2.extras import execute_values
import numpy as np

DB_CONFIG = {
    "dbname": "cbde_l1",
    "user": "postgres",
    "password": "postgrespassword",  # Canvia-ho per la teva contrasenya
    "host": "localhost",
    "port": 5432
}

with open("../bookcorpus_sentences.json", "r", encoding="utf-8") as f:
    sentences = json.load(f)

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Taula relacional pura
cur.execute("DROP TABLE IF EXISTS corpus;")
cur.execute("""
    CREATE TABLE corpus (
        id SERIAL PRIMARY KEY,
        sentence TEXT NOT NULL
    );
""")
conn.commit()

# Inserció per lots
chunk_size = 500
chunks = [sentences[i:i + chunk_size] for i in range(0, len(sentences), chunk_size)]
insert_times = []

for chunk in chunks:
    data = [(s,) for s in chunk]
    t0 = time.perf_counter()
    execute_values(cur, "INSERT INTO corpus (sentence) VALUES %s;", data)
    conn.commit()
    t1 = time.perf_counter()
    insert_times.append(t1 - t0)

cur.close()
conn.close()

print("\n--- PERFORMANCE ---")
print(f"Min: {np.min(insert_times):.6f} s")
print(f"Max: {np.max(insert_times):.6f} s")
print(f"Mean: {np.mean(insert_times):.6f} s")
print(f"Standard Deviation: {np.std(insert_times):.6f} s")
print(f"Total Time: {np.sum(insert_times):.4f} s")