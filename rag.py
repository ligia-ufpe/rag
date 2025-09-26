import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# -----------------------------
# Configurações iniciais
# -----------------------------
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("❌ Variável de ambiente OPENROUTER_API_KEY não encontrada no .env")

# -----------------------------
# Instancia o LLM (via OpenRouter)
# -----------------------------
llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",   # no site do OpenRouter tem outros modelos também
    api_key=api_key,                              
    base_url="https://openrouter.ai/api/v1",      # obrigatório
    temperature=0,
    max_tokens=512,
)

# -----------------------------
# Embeddings open source (Hugging Face)
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Carregamento e indexação do PDF
# -----------------------------
pdf_path = "receitas_bolos.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"❌ Arquivo PDF não encontrado: {pdf_path}")

loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = splitter.split_documents(docs)

vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Prompt fixo
# -----------------------------
prompt = ChatPromptTemplate.from_template(
    "Você é um assistente especializado em receitas de bolos caseiros. "
    "Responda exclusivamente com base no documento de referência fornecido. "
    "Siga estas regras:\n\n"
    "- Responda sempre em português brasileiro, de forma clara, didática e acolhedora.\n"
    "- Explique ingredientes, quantidades e modo de preparo quando solicitados.\n"
    "- Se perguntarem sobre substituições, sugira apenas se houver algo parecido no documento.\n"
    "- Se pedirem dicas de preparo, traga o que o guia sugere ou a lógica baseada nos ingredientes descritos.\n"
    "- Nunca invente receitas ou ingredientes que não estejam no material.\n"
    "- Se a pergunta não tiver relação com receitas de bolos, responda educadamente que não pode responder.\n\n"
    "Documento de referência:\n{context}\n\n"
    "Pergunta: {question}"
)

# -----------------------------
# Função principal de resposta
# -----------------------------
def responder_pergunta(pergunta: str) -> str:
    """Recebe uma pergunta e retorna a resposta do LLM baseada no PDF indexado."""
    print("DEBUG: Pergunta recebida:", pergunta)

    docs = retriever.invoke(pergunta)
    print(f"DEBUG: {len(docs)} documentos recuperados")
    for i, d in enumerate(docs, 1):
        print(f"--- Doc {i} ---")
        print(d.page_content[:200], "...")

    contexto = "\n\n".join([d.page_content for d in docs])
    mensagens = prompt.format_messages(context=contexto, question=pergunta)

    try:
        resposta = llm.invoke(mensagens)
        print("DEBUG: Resposta bruta:", resposta)
    except Exception as e:
        print("❌ Erro ao chamar o LLM:", e)
        raise e

    return resposta.content if hasattr(resposta, "content") else str(resposta)