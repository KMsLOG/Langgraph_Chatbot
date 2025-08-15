import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_chroma import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL_NAME, RERANKER_MODEL_NAME

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