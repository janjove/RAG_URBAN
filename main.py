from pypdf import PdfReader
from process_pdf import processar_pdf
from chunk import chunk_text

## retorna el text netejat i desat en JSON
processar_pdf("prova.pdf")



# Exemple amb una pàgina de JSON
import json
with open("doc_netejat.json", "r", encoding="utf-8") as f:
    dades = json.load(f)

tots_chunks = []
for entry in dades:
    pagina = entry["pagina"]
    text = entry["text_netejat"]
    # Chunking per pàgina
    chunks = chunk_text(text, max_words=300, overlap=50)
    for i, ch in enumerate(chunks):
        tots_chunks.append({
            "pagina": pagina,
            "chunk_id": f"{pagina}_{i}",
            "text": ch
        })

print(f"S'han generat {len(tots_chunks)} chunks")


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Carrega model d’embeddings (lleuger i ràpid)
model = SentenceTransformer("intfloat/multilingual-e5-base")

# Carrega els chunks
with open("doc_netejat.json", "r", encoding="utf-8") as f:
    dades = json.load(f)

tots_chunks = []
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

# Calcular embeddings (llista de vectors)
print("Calculant embeddings...")
texts = [c["text"] for c in tots_chunks]
embeddings = model.encode(
    texts,
    convert_to_numpy=True,
    show_progress_bar=True
)

# Crear índex FAISS
d = embeddings.shape[1]  # dimensió del vector
index = faiss.IndexFlatL2(d)
index.add(embeddings)  # afegeix tots els vectors

# Exemple de consulta
query = "Quins són els punts principals de la introducció?"
query_embedding = model.encode([query], convert_to_numpy=True)
D, I = index.search(query_embedding, k=3)  # retorna 3 chunks més similars

print("Resultats de la cerca:")
for idx in I[0]:
    print(f"- (pàgina {tots_chunks[idx]['pagina']}) {tots_chunks[idx]['text'][:200]}...")
