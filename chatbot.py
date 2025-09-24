import os
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
#import google.generativeai as genai

# from utils.pdf_loader import load_manual
# from utils.rerank import rerank
# from utils.embedding_index import build_embedding_index, load_embedding_index


DATA_DIR = Path("data")
PDF_FILE = DATA_DIR / "manual.pdf"
INDEX_DIR = DATA_DIR / "faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Load env and API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("Set GOOGLE_API_KEY in .env")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.2,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

def get_text_from_pdf(path: Path):
    reader = PdfReader(str(path))
    pages = []
    metas = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        if txt.strip():
            pages.append(txt.strip())
            metas.append({"page": i + 1})
    return pages, metas

def get_or_build_index():
    if INDEX_DIR.exists():
        return FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    # build fresh index
    texts, metas = get_text_from_pdf(PDF_FILE)
    vs = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metas)
    vs.save_local(str(INDEX_DIR))
    return vs

def make_chain(retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based only on the following context:\n\n{context}"),
        ("human", "{question}")
    ])
    return (
        {
            "context": lambda x: "\n\n".join([d.page_content for d in retriever.invoke(x["question"])]),
            "question": lambda x: x["question"],
        }
        | prompt | llm | StrOutputParser())


def main():
    vs = get_or_build_index()
    retriever = vs.as_retriever(search_kwargs={"k": 5})
    qa_chain = make_chain(retriever)

    print("Chatbot ready. Type exit to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        try:
            resp = qa_chain.invoke({"question": q})
            print("\nBot:", resp)
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
