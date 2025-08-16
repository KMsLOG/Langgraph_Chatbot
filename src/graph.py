from typing import Literal
from typing import List, TypedDict, Annotated
from operator import add
from textwrap import dedent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tools import personal_law_search, labor_law_search, housing_law_search, web_search
from agents import create_rag_agent, llm

personal_law_agent = create_rag_agent(personal_law_search, "개인정보보호법 전문가", "개인정보 보호법")
labor_law_agent = create_rag_agent(labor_law_search, "근로기준법 전문가", "근로기준법")
housing_law_agent = create_rag_agent(housing_law_search, "주택임대차보호법 전문가", "주택임대차보호법")
web_search_agent = create_rag_agent(web_search, "인터넷 정보 검색 전문가", "웹 검색")

class ResearchAgentState(TypedDict):
    """리서치 에이전트 그래프 상태"""
    question: str
    answers: Annotated[List[str], add]
    final_answer: str
    datasources: List[str]
    
class ToolSelector(BaseModel):
    """tool 셀렉터 - 단일 도구 선택"""
    tool: Literal["search_personal", "search_labor", "search_housing", "search_web"]

class ToolSelectors(BaseModel):
    """tool 셀렉터 - 다중 도구 선택"""
    tools: List[ToolSelector]

structured_llm_router = llm.with_structured_output(ToolSelectors)
route_prompt = ChatPromptTemplate.from_messages([
    ("system", dedent("""당신은 사용자 질문을 적절한 도구로 라우팅하는 전문가입니다.
    - 개인정보 보호법 조항 관련 질문은 'search_personal'을 사용하세요.
    - 근로기준법 조항 관련 질문은 'search_labor'를 사용하세요.
    - 주택임대차보호법 조항 관련 질문은 'search_housing'을 사용하세요.
    - 그 외 모든 정보, 최신 데이터, 법률과 관련되었지만 특정 조항이 아닌 질문은 'search_web'을 사용하세요.
    - 질문이 법률과 관련되었지만 특정 조항에 대한 질문이 아닌 경우, 관련 법률 검색 도구와 웹 검색 도구를 모두 포함하세요.""")),
    ("human", "{question}"),
])
question_router = route_prompt | structured_llm_router

def analyze_question(state: ResearchAgentState):
    """질문 분석"""
    print("1 - 질문 분석")
    result = question_router.invoke({"question": state["question"]})
    return {"datasources": [tool.tool for tool in result.tools]}

def run_personal_rag(state: ResearchAgentState):
    """개인정보 RAG 실행"""
    print("2.1 - 개인정보 RAG 실행")
    result = personal_law_agent.invoke({"question": state["question"]})
    return {"answers": [result["node_answer"]]}

def run_labor_rag(state: ResearchAgentState):
    """근로 RAG 실행"""
    print("2.2 - 근로기준 RAG 실행")
    result = labor_law_agent.invoke({"question": state["question"]})
    return {"answers": [result["node_answer"]]}

def run_housing_rag(state: ResearchAgentState):
    """주택 RAG 실행"""
    print("2.3 - 주택 RAG 실행")
    result = housing_law_agent.invoke({"question": state["question"]})
    return {"answers": [result["node_answer"]]}

def run_web_search(state: ResearchAgentState):
    """웹 검색 실행"""
    print("2.4 - 웹검색 실행")
    result = web_search_agent.invoke({"question": state["question"]})
    return {"answers": [result["node_answer"]]}

def generate_final_answer(state: ResearchAgentState):
    """답변 생성"""
    print("3 - 답변 생성")
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", dedent("""당신은 여러 출처의 정보를 종합하여 명확하고 간결한 최종 답변을 생성하는 AI 어시스턴트입니다.
        - 제공된 문서의 정보만을 사용하세요.
        - 각 정보의 출처를 문장 끝에 명확히 표기하세요. 예: (출처: 법률명 제X조) 또는 (출처: 웹사이트명, URL)""")),
        ("human", "다음 정보를 바탕으로 질문에 답변하세요:\n\n[정보]\n{documents}\n\n[질문]\n{question}"),
    ])
    rag_chain = rag_prompt | llm
    documents_text = "\n\n".join(state["answers"])
    final_answer = rag_chain.invoke({"documents": documents_text, "question": state["question"]})
    return {"final_answer": final_answer.content}
    
def route_datasources(state: ResearchAgentState):
    """
    여러 데이터소스가 선택된 경우 병렬 실행을 위해 리스트를 반환
    """
    return state['datasources']

def build_graph():
    """그래프 빌드"""
    workflow = StateGraph(ResearchAgentState)

    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("search_personal", run_personal_rag)
    workflow.add_node("search_labor", run_labor_rag)
    workflow.add_node("search_housing", run_housing_rag)
    workflow.add_node("search_web", run_web_search)
    workflow.add_node("generate_answer", generate_final_answer)

    workflow.set_entry_point("analyze_question")

    workflow.add_conditional_edges(
        "analyze_question",
        route_datasources,
        {
            "search_personal": "search_personal",
            "search_labor": "search_labor",
            "search_housing": "search_housing",
            "search_web": "search_web",
        }
    )

    for node in ["search_personal", "search_labor", "search_housing", "search_web"]:
        workflow.add_edge(node, "generate_answer")

    workflow.add_edge("generate_answer", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

legal_rag_agent = build_graph()
