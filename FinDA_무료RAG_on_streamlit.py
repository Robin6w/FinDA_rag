import tempfile
import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# --------------------------------------------------
# Streamlit 기본 설정
# --------------------------------------------------
st.set_page_config(page_title="100% Free Korean RAG", layout="wide")
st.title("📄 100% 무료 한국어 RAG 챗봇 (Qwen + FAISS)")

# --------------------------------------------------
# 모델 설정 (한글 안정 조합)
# --------------------------------------------------
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

TOP_K = 1
MAX_CONTEXT_CHARS = 700
MAX_NEW_TOKENS = 256

# --------------------------------------------------
# 캐시: Embedding / LLM
# --------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

@st.cache_resource
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,      # ✅ 메모리 절감 핵심
        low_cpu_mem_usage=True,         # ✅ 로딩 메모리 절감
        device_map="cpu",               # ✅ Cloud는 CPU
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128,             # ✅ 256 → 128로 내리기
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.05,
        return_full_text=False,
    )
    return gen, tokenizer

# --------------------------------------------------
# 유틸
# --------------------------------------------------
def format_docs(docs, max_chars):
    text = "\n\n".join(d.page_content for d in docs if d.page_content)
    return text[:max_chars]

# --------------------------------------------------
# 세션 상태
# --------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------------------------------------------
# PDF 업로드
# --------------------------------------------------
uploaded = st.file_uploader("PDF 파일 업로드", type=["pdf"])

if uploaded:
    with st.spinner("PDF 분석 및 인덱싱 중..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.getvalue())
            pdf_path = tmp.name

        loader = PDFPlumberLoader(pdf_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80
        )
        splits = splitter.split_documents(docs)

        embeddings = load_embeddings()
        st.session_state.vectorstore = FAISS.from_documents(splits, embeddings)

    st.success(f"인덱싱 완료 (chunks: {len(splits)})")

# --------------------------------------------------
# 채팅 UI
# --------------------------------------------------
st.subheader("💬 문서 기반 Q&A")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_q = st.chat_input("질문을 입력하세요")

if user_q:
    st.session_state.chat_history.append(
        {"role": "user", "content": user_q}
    )
    with st.chat_message("user"):
        st.write(user_q)

    if st.session_state.vectorstore is None:
        answer = "먼저 PDF를 업로드하세요."
    else:
        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )
        docs = retriever.invoke(user_q)
        context = format_docs(docs, MAX_CONTEXT_CHARS)

        gen, tok = load_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "당신은 업로드된 PDF 문서의 내용만 근거로 "
                    "한국어로 답하는 도우미입니다. "
                    "문맥에 없으면 '문맥에 근거해 알 수 없습니다.'라고 답하세요."
                )
            },
            {
                "role": "user",
                "content": f"[문맥]\n{context}\n\n[질문]\n{user_q}"
            }
        ]

        prompt = tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        out = gen(prompt)
        answer = out[0]["generated_text"].strip()

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

