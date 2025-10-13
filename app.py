import streamlit as st
import json
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from chunk import chunk_text
import os
from dotenv import load_dotenv

# -----------------------------
# CONFIGURACIÓ
# -----------------------------
INDEX_PATH = "index_document88.faiss"
CHUNKS_PATH = "doc_netejat.json"
MODEL_EMB = "intfloat/multilingual-e5-base"
OPENAI_MODEL = "gpt-4o-mini"
K = 5

# Carrega clau API
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# CÀRREGA DE RECURSOS
# -----------------------------
@st.cache_resource
def carregar_model_i_index():
    st.info("Carregant model d'embeddings i índex FAISS...")
    emb_model = SentenceTransformer(MODEL_EMB)
    index = faiss.read_index(INDEX_PATH)

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
    return emb_model, index, tots_chunks

emb_model, index, tots_chunks = carregar_model_i_index()

# -----------------------------
# FUNCIÓ DE CONSULTA
# -----------------------------
def consulta(query):
    query_emb = emb_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k=K)

    context = []
    for idx in I[0]:
        context.append(f"[Pàgina {tots_chunks[idx]['pagina']}] {tots_chunks[idx]['text']}")

    prompt = f"""
Respon en català de manera clara i sintètica, basant-te exclusivament en el context donat.
Si no hi ha prou informació, indica-ho.

Pregunta: {query}

Context:
{chr(10).join(context)}
"""

    resposta = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "Ets un assistent que respon preguntes sobre el document donat."},
            {"role": "user", "content": prompt}
        ]
    )
    return resposta.choices[0].message.content, context

# -----------------------------
# INTERFÍCIE STREAMLIT
# -----------------------------
st.set_page_config(page_title="RAG amb FAISS", layout="wide")
st.title("💬 Sistema RAG amb FAISS + GPT-4o-mini")

query = st.text_input("Introdueix la teva pregunta:", "")

if st.button("Cercar resposta") and query.strip():
    with st.spinner("Buscant informació..."):
        resposta, cites = consulta(query)

    st.subheader("📘 Resposta")
    st.write(resposta)

    with st.expander("📄 Fragments utilitzats"):
        for i, c in enumerate(cites, 1):
            st.markdown(f"**Fragment {i}:** {c}")

st.markdown("---")
st.caption("Construït per Jan · Basat en FAISS, SentenceTransformers i GPT-4o-mini")
