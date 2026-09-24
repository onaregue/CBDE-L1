import time
import psycopg2
from psycopg2.extras import execute_batch
from sentence_transformers import SentenceTransformer
import numpy as np

DB_CONFIG = {
    "dbname": "cbde_l1",
    "user": "postgres",
    "password": "postgrespassword",  # Canvia-ho per la teva contrasenya
    "host": "localhost",
    "port": 5432
}

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

# Afegim columna d'arrays natius de PostgreSQL
cur.execute("ALTER TABLE corpus ADD COLUMN IF NOT EXISTS embedding FLOAT8[];")
conn.commit()

cur.execute("SELECT id, sentence FROM corpus ORDER BY id;")
rows = cur.fetchall()
ids = [r[0] for r in rows]
sentences = [r[1] for r in rows]

embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True)

batch_size = 500
update_data = [(emb.tolist(), row_id) for emb, row_id in zip(embeddings, ids)]
batches = [update_data[i:i + batch_size] for i in range(0, len(update_data), batch_size)]
update_times = []

for batch in batches:
    t0 = time.perf_counter()
    execute_batch(cur, "UPDATE corpus SET embedding = %s WHERE id = %s;", batch)
    conn.commit()
    t1 = time.perf_counter()
    update_times.append(t1 - t0)

cur.close()
conn.close()

print("\n--- MÈTRIQUES INSERCIÓ EMBEDDINGS [P1] ---")
print(f"Mínim: {np.min(update_times):.6f} s")
print(f"Màxim: {np.max(update_times):.6f} s")
print(f"Mitjana: {np.mean(update_times):.6f} s")
print(f"Desviació Estàndard: {np.std(update_times):.6f} s")
print(f"Temps total: {np.sum(update_times):.4f} s")