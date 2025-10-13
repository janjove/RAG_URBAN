import json
import faiss
from sentence_transformers import SentenceTransformer

# 1. Carrega el model d’embeddings (ha de ser el mateix que vas usar abans!)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# Si vols suport multilingüe millor, pots fer:
# model = SentenceTransformer("intfloat/multilingual-e5-base")

# 2. Carrega els chunks des del JSON (metadades)
with open("doc_netejat.json", "r", encoding="utf-8") as f:
    dades = json.load(f)

tots_chunks = []
from text_chunking import chunk_text
for entry in dades:
    pagina = entry["pagina"]
    text = entry["text_netejat"]
    chunks = chunk_text(text, max_words=300, overlap=50)
    for i, ch in enumerate(chunks):
        tots_chunks.append({
            "pagina": pagina,
            "chunk_id": f"{pagina}_{i}",
            "text": ch
        })

# 3. Carrega l’índex FAISS que ja havies desat
index = faiss.read_index("index_document88.faiss")

# 4. Consulta d’exemple
query = "Quin és l’objectiu principal del document?"
query_emb = model.encode([query], convert_to_numpy=True)

# 5. Cerca dels k més propers
D, I = index.search(query_emb, k=5)

# 6. Mostra els fragments més rellevants
print("Resultats de la consulta:\n")
for idx in I[0]:
    print(f"[Pàgina {tots_chunks[idx]['pagina']}, Chunk {tots_chunks[idx]['chunk_id']}]")
    print(tots_chunks[idx]["text"])
    print("-" * 80)
