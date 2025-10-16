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

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# CÀRREGA DE MODELS I DADES
# -----------------------------
@st.cache_resource
def carregar_model_i_index():
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
# FUNCIONS AUXILIARS
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
# INTERFÍCIE STREAMLIT (XAT)
# -----------------------------
st.set_page_config(page_title="Xat RAG amb FAISS", layout="wide")
st.title("💬 Xat amb el teu document (RAG)")

# Inicia la memòria de xat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostra tots els missatges anteriors
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input de l’usuari
if query := st.chat_input("Escriu la teva pregunta sobre el document..."):
    # Mostra el missatge de l’usuari
    st.chat_message("user").markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Genera la resposta amb RAG
    with st.spinner("Pensant..."):
        resposta, cites = consulta(query)

    # Mostra la resposta
    with st.chat_message("assistant"):
        st.markdown(resposta)

        with st.expander("📄 Fragments utilitzats"):
            for i, c in enumerate(cites, 1):
                st.markdown(f"**Fragment {i}:** {c[:500]}...")

    st.session_state.messages.append({"role": "assistant", "content": resposta})
