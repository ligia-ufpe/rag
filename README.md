# 📘 RAG com LangChain e OpenRouter

Este projeto implementa um sistema de **RAG (Retrieval-Augmented Generation)** utilizando [LangChain](https://www.langchain.com/), **FAISS** para indexação vetorial, **embeddings do Hugging Face** e **modelos LLM via OpenRouter**.  

O objetivo é responder perguntas de forma contextualizada com base em um documento PDF.  
Para demonstração, foi utilizado um PDF contendo receitas de bolos caseiros, servindo como exemplo de aplicação prática.

---

## ⚙️ Funcionalidades

- 📂 **Carregamento de documentos** no formato PDF.  
- ✂️ **Divisão inteligente** do texto em *chunks* com sobreposição para manter o contexto.  
- 🧩 **Geração de embeddings vetoriais** utilizando modelos da Hugging Face.  
- 🔎 **Indexação com FAISS** para busca semântica rápida e eficiente.  
- 🤖 **Integração com modelos LLM** via OpenRouter.  
- ✅ **Respostas sempre baseadas no documento fornecido**, evitando alucinações.  
---

## 📂 Estrutura do Projeto
```
├── app.py              # Interface com Streamlit (se usada)
├── rag.py              # Núcleo do RAG: carregamento, indexação e resposta
├── receitas_bolos.pdf  # Documento de referência
├── requirements.txt    # Dependências do projeto
└── README.md           # Este arquivo
```
---

## 🚀 Como Executar

### 1. Instale o Conda

Se ainda não tiver o Conda, você pode instalar o [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (mais leve) ou o [Anaconda](https://www.anaconda.com/download).  

Exemplo de instalação do **Miniconda** no Linux/macOS:  
# Baixe o instalador (Linux)
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
# Ou no macOS (Apple Silicon)
```
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
```
# Instale
```
bash Miniconda3-latest-*.sh
```
No Windows, basta baixar o instalador gráfico do site oficial e seguir os passos.

Depois da instalação, reinicie o terminal e verifique:
```
conda --version
```

⸻

2. Crie e ative o ambiente virtual
```
conda create -n rag-env python=3.12 -y
conda activate rag-env
```

⸻

3. Clone o repositório
```
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

⸻

4. Instale as dependências
```
pip install -r requirements.txt
```

⸻

5. Configure as variáveis de ambiente

Crie um arquivo .env na raiz do projeto:
```
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxxxxxx
```

⸻

6. Execute a aplicação

Se estiver usando apenas o núcleo RAG:
```
python rag.py
```
Se tiver interface Streamlit:
```
streamlit run app.py
```

⸻

🧠 Principais Componentes

🔑 Carregamento e indexação do PDF
```
loader = PyMuPDFLoader("receitas_bolos.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

```
🔎 Recuperação e resposta
```
def responder_pergunta(pergunta: str) -> str:
    docs = retriever.invoke(pergunta)
    contexto = "\n\n".join([d.page_content for d in docs])
    mensagens = prompt.format_messages(context=contexto, question=pergunta)
    resposta = llm.invoke(mensagens)
    return resposta.content
```

⸻

##📚 Tecnologias Utilizadas
- **LangChain** – Orquestração do fluxo RAG.  
- **FAISS** – Indexação e busca vetorial semântica.  
- **Hugging Face Sentence Transformers** – Criação de embeddings.  
- **OpenRouter** – Acesso a modelos LLM.  
- **Streamlit** – Interface interativa (opcional).  
- **Conda** – Gerenciamento de ambientes.  

---

##💡 Observações
- As respostas são sempre extraídas do **PDF fornecido**.  
- Caso a pergunta não esteja relacionada ao documento, o modelo responde educadamente que não pode responder.  
- O modelo padrão é **mistralai/mistral-7b-instruct:free**, mas pode ser substituído por outros disponíveis no catálogo do **OpenRouter**.  
⸻

**Autor:** [Gabriel W. A. Matias](https://www.linkedin.com/in/gabriel-w-a-matias-a9913a210/)
