import streamlit as st
from rag import responder_pergunta  # importa a função do rag.py

st.set_page_config(page_title="RAG", layout="centered")
st.title("RAG - Retrieval Augmented Generation")
st.caption("Pergunte sobre receitas de bolos caseiros 🍰")

pergunta = st.text_input("Digite sua pergunta:")

if st.button("Enviar"):
    if not pergunta.strip():
        st.warning("⚠️ Digite uma pergunta primeiro!")
    else:
        with st.spinner("Buscando resposta..."):
            try:
                resposta = responder_pergunta(pergunta)
                st.markdown("## 📣 Resposta:")
                st.success(resposta)
            except Exception as e:
                st.error(f"Erro ao gerar resposta: {e}")