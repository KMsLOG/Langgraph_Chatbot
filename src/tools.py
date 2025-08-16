import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import TavilySearchAPIRetriever
from typing import List
from langchain_core.documents import Document
from langchain_core.tools import tool
from config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

embeddings_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

rerank_model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL_NAME)
compressor = CrossEncoderReranker(model=rerank_model, top_n=2)

def create_retriever(collection_name:str):
    """컬렉션 이름에 맞는 검색기 생성 함수"""
    if not os.path.exists(CHROMA_DB_PATH):
        return None
    db = Chroma(
        embedding_function=embeddings_model,
        collection_name=collection_name,
        persist_directory=CHROMA_DB_PATH,
    )
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=db.as_retriever(search_kwargs={"k": 5}),
    )

# 각 검색기 생성
personal_retriever = create_retriever("personal_law")
labor_retriever = create_retriever("labor_law")
housing_retriever = create_retriever("housing_law")

# Tavily를 이용한 웹검색기
web_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=TavilySearchAPIRetriever(k=10),
)

db_error_msg = "벡터DB가 존재하지 않습니다."

# tool 정의
@tool
def personal_law_search(query: str) -> List[Document]:
    """개인정보보호법 법률 조항을 검색하는 툴"""
    if not personal_retriever:
        return [Document(page_content=db_error_msg)]
    docs = personal_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def labor_law_search(query: str) -> List[Document]:
    """근로기준법 법률 조항을 검색하는 툴"""
    if not labor_retriever:
        return [Document(page_content=db_error_msg)]
    docs = labor_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def housing_law_search(query: str) -> List[Document]:
    """주택임대차보호법 법률 조항을 검색하는 툴"""
    if not housing_retriever:
        return [Document(page_content=db_error_msg)]
    docs = housing_retriever.invoke(query)
    return docs if docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def web_search(query: str) -> List[Document]:
    """데이터베이스에 없는 정보 또는 최신 정보를 웹에서 검색하는 툴"""
    docs = web_retriever.invoke(query)
    formatted_docs = [
        Document(
            page_content=f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>',
            metadata={"source": "web search", "url": doc.metadata["source"]},
        ) for doc in docs
    ]
    return formatted_docs if formatted_docs else [Document(page_content="관련 정보를 찾을 수 없습니다.")]

law_tools = [personal_law_search, labor_law_search, housing_law_search, web_search]
