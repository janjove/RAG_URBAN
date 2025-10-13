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
    chunks = chunk_text(text, max_words=500, overlap=150)
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
texts = [c["text"] for c in tots_chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# Crear índex FAISS
d = embeddings.shape[1]  # dimensió del vector
index = faiss.IndexFlatL2(d)
index.add(embeddings)  # afegeix tots els vectors

# Desa a disc
faiss.write_index(index, "index_document88.faiss")

