import json
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
# Reconstrueix tots els chunks (ha
# n d'estar en el mateix ordre que FAISS)
from chunk import chunk_text
import os
from dotenv import load_dotenv
# -----------------------------
# Configuració
# -----------------------------
INDEX_PATH = "index_document88.faiss"
CHUNKS_PATH = "doc_netejat.json"
MODEL_EMB = "intfloat/multilingual-e5-base"   # mateix model que a l'index
OPENAI_MODEL = "gpt-4o-mini"
K = 5   # nombre de fragments a recuperar
# Carrega la clau API des de les variables d'entorn
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


print("API KEY:", OPENAI_API_KEY)

# -----------------------------
# Carrega embeddings i FAISS
# -----------------------------
print("Carregant model d'embeddings...")
emb_model = SentenceTransformer(MODEL_EMB)

print("Carregant chunks...")
with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
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

print("Carregant índex FAISS...")
index = faiss.read_index(INDEX_PATH)

# -----------------------------
# Funció de consulta
# -----------------------------
def consulta(query):
    # 1. Encodeja la query
    query_emb = emb_model.encode([query], convert_to_numpy=True)

    # 2. Recupera k chunks més propers
    D, I = index.search(query_emb, k=K)
    context = []
    for idx in I[0]:
        context.append(f"[Pàgina {tots_chunks[idx]['pagina']}] {tots_chunks[idx]['text']}")

    # 3. Construeix el prompt
    prompt = f"""
Respon en català de manera clara i sintètica, basant-te exclusivament en el context donat.
Si no hi ha prou informació, indica-ho.

Pregunta: {query}

Context:
{chr(10).join(context)}
"""

    # 4. Crida a OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resposta = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Ets un assistent que respon preguntes sobre el document donat."},
            {"role": "user", "content": prompt}
        ]
    )

    # 5. Retorna resultat
    return resposta.choices[0].message.content, context

# -----------------------------
# Executa consulta
# -----------------------------
if __name__ == "__main__":
    preguntes = [
        "Quin any es va encarregar l’estudi de pla d’usos comercials de Sant Andreu de Llavaneres?",
        "Quins tres àmbits d’actuació proposava l’estudi de 2010?",
        "Quan es va aprovar definitivament el POUM del municipi?",
        "Quin és l’objectiu principal del Polígon d’Actuació Urbanística Les Alzines?",
        "Quina era la població de Sant Andreu de Llavaneres l’any 2024?",
        "Quin és el valor de la renda familiar disponible bruta per habitant al municipi?",
        "Quin és el sector econòmic predominant a Sant Andreu de Llavaneres segons el document?",
        "Quants establiments comercials actius hi ha al municipi segons el cens del 2025?",
        "Quins sectors comercials presenten un dèficit d’oferta al municipi segons la balança comercial?",
        "Quina és la relació entre renda mitjana i capacitat de despesa al municipi?"
    ]

    output_path = "resultats_RAG_2.txt"

    with open(output_path, "w", encoding="utf-8") as out:
        for i, q in enumerate(preguntes, start=1):
            print(f"\n🧠 Pregunta {i}: {q}")
            resposta, cites = consulta(q)

            # Mostra per pantalla
            print("\n--- Resposta ---")
            print(resposta)
            print("\n--- Fragments utilitzats ---")
            for c in cites:
                print("-", c[:200], "...")

            # Escriu al fitxer
            out.write(f"\n🧠 Pregunta {i}: {q}\n")
            out.write(f"--- Resposta ---\n{resposta}\n")
            out.write("--- Fragments utilitzats ---\n")
            for c in cites:
                out.write(f"- {c[:500]} ...\n")
            out.write("-" * 80 + "\n")

    print(f"\n✅ Resultats guardats a: {output_path}")

