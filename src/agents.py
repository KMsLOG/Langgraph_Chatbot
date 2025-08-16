from typing import List, TypedDict, Annotated, Optional, Literal
from operator import add
from textwrap import dedent
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from config import LLM_MODEL_NAME

class InformationStrip(BaseModel):
    """추출된 정보에 대한 내용/출처/관련성 점수"""
    content: str = Field(..., description="추출된 정보 내용")
    source: str = Field(..., description="정보의 출처(법률 조항/URL 등)")
    relevance_score: float = Field(..., ge=0, le=1, description="질의에 대한 관련성 점수 (0~1)")
    faithfulness_score: float = Field(..., ge=0, le=1, description="답변의 충실성 점수 (0~1)")

class ExtractedInformation(BaseModel):
    """추출된 정보들"""
    strips: List[InformationStrip]
    query_relevance: float = Field(..., ge=0, le=1, description="질의에 대한 답변 가능성 점수 (0~1)")

class RefinedQuestion(BaseModel):
    """개선된 질문/이유"""
    question_refined: str = Field(..., description="개선된 질문")
    reason: str = Field(..., description="이유")

class CorrectiveRagState(TypedDict):
    """RAG 상태"""
    question: str
    rewritten_query: Optional[str]
    documents: List[Document]
    extracted_info: Optional[List[InformationStrip]]
    generation: str
    num_generations: int
    node_answer: Optional[str]

llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0, streaming=True)


def create_rag_agent(search_tool: callable, expert_name: str, expert_domain: str):
    """RAG 에이전트 그래프를 생성하는 팩토리 함수"""

    def retrieve_documents(state: CorrectiveRagState) -> CorrectiveRagState:
        """검색기로 검색한 내용 반환"""
        print(f"[{expert_name}] 문서 검색")
        query = state.get("rewritten_query") or state["question"]
        docs = search_tool.invoke(query)
        return {"documents": docs}

    def extract_and_evaluate(state: CorrectiveRagState) -> CorrectiveRagState:
        """추출 및 평가"""
        print(f"[{expert_name}] 정보 추출 및 평가")
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(f"""당신은 {expert_name}입니다. 주어진 문서에서 질문과 관련된 주요 사실과 정보를 3~5개 정도 추출하세요. 
            각 정보에 대해 관련성과 충실성을 0에서 1 사이의 점수로 평가하고, 최종적으로 질문에 대한 답변 가능성 점수를 평가하세요.""")),
            ("human", "[질문]\n{question}\n\n[문서 내용]\n{document_content}")
        ])
        extract_llm = llm.with_structured_output(ExtractedInformation)
        extraction_chain = extract_prompt | extract_llm
        
        extracted_strips = []
        for doc in state["documents"]:
            extracted_data = extraction_chain.invoke({
                "question": state["question"], "document_content": doc.page_content
            })
            if extracted_data.query_relevance < 0.8: continue
            for strip in extracted_data.strips:
                if strip.relevance_score > 0.7 and strip.faithfulness_score > 0.7:
                    strip.source = doc.metadata.get("source", "N/A")
                    extracted_strips.append(strip)
        
        return {"extracted_info": extracted_strips, "num_generations": state.get("num_generations", 0) + 1}

    def rewrite_query(state: CorrectiveRagState) -> CorrectiveRagState:
        """쿼리 재작성"""
        print(f"[{expert_name}] 쿼리 재작성")
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(f"""당신은 {expert_name}입니다. 주어진 질문과 추출된 정보를 바탕으로 더 나은 정보를 찾기 위해 검색 쿼리를 개선해주세요. 
            가장 효과적일 것 같은 쿼리 하나와 그 이유를 제시하세요.""")),
            ("human", "원래 질문: {question}\n\n추출된 정보:\n{extracted_info}")
        ])
        rewrite_llm = llm.with_structured_output(RefinedQuestion)
        rewrite_chain = rewrite_prompt | rewrite_llm
        extracted_info_str = "\n".join([strip.content for strip in state["extracted_info"]])
        response = rewrite_chain.invoke({"question": state["question"], "extracted_info": extracted_info_str})
        return {"rewritten_query": response.question_refined}

    def generate_answer(state: CorrectiveRagState) -> CorrectiveRagState:
        """답변 생성"""
        print(f"[{expert_name}] 답변 생성")
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", dedent(f"""당신은 {expert_name}입니다. 주어진 질문과 정보를 바탕으로 답변을 생성해주세요. 
            답변은 마크다운 형식으로 작성하며, 각 정보의 출처를 명확히 표시해야 합니다. 
            예: (출처: {expert_domain} 제15조)""")),
            ("human", "질문: {question}\n\n추출된 정보:\n{extracted_info}")
        ])
        extracted_info_str = "\n".join([f"내용: {s.content}\n출처: {s.source}" for s in state["extracted_info"]])
        answer = answer_prompt | llm
        node_answer = answer.invoke({"question": state["question"], "extracted_info": extracted_info_str})
        return {"node_answer": node_answer.content}

    def should_continue(state: CorrectiveRagState) -> Literal["continue", "end"]:
        """진행 여부 결정"""
        if state["num_generations"] >= 2 or len(state.get("extracted_info", [])) >= 1:
            return "end"
        return "continue"

    workflow = StateGraph(CorrectiveRagState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("extract_and_evaluate", extract_and_evaluate)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("generate_answer", generate_answer)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "extract_and_evaluate")
    workflow.add_conditional_edges(
        "extract_and_evaluate",
        should_continue,
        {"continue": "rewrite_query", "end": "generate_answer"}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()
