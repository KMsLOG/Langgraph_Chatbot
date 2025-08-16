from logger_config import setup_logger
import logging
import gradio as gr
import uuid
from typing import List, Tuple
from graph import legal_rag_agent

setup_logger()
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self):
        self.thread_id = str(uuid.uuid4())

    def _get_config(self):
        return {"configurable": {"thread_id": self.thread_id}}

    def _get_current_state(self):
        try:
            config = self._get_config()
            state = legal_rag_agent.get_state(config)
            return state.values if hasattr(state, 'values') else state
        except Exception as e:
            print(f"상태 가져오기 오류: {e}")
            return None

    def chat(self, message: str, history: List[Tuple[str, str]]) -> str:
        print(f"Thread ID: {self.thread_id}, User Message: {message}")
        config = self._get_config()

        try:
            result = legal_rag_agent.invoke(
                {"question": message, "answers": []}, 
                config
            )
            
            if isinstance(result, dict):
                final_answer = result.get("final_answer", "답변을 찾을 수 없습니다.")
            else:
                current_state = self._get_current_state()
                if current_state and isinstance(current_state, dict):
                    final_answer = current_state.get("final_answer", "답변을 찾을 수 없습니다.")
                else:
                    final_answer = "답변을 처리할 수 없습니다."
            
            return final_answer
            
        except Exception as e:
            print(f"질문 처리 오류: {e}")
            return "질문 처리 중 오류가 발생했습니다. 다시 시도해주세요."

def create_chatbot_interface():
    chatbot = ChatBot()
    
    example_questions = [
        "사업장에서 CCTV를 설치할 때 주의해야 할 법적 사항은 무엇인가요?",
        "전월세 계약 갱신 요구권의 행사 기간과 조건은 어떻게 되나요?",
        "개인정보 유출 시 기업이 취해야 할 법적 조치는 무엇인가요?",
    ]

    with gr.Blocks(theme=gr.themes.Soft(), title="생활법률 AI 어시스턴트") as demo:
        gr.Markdown("<h1>생활법률 AI 어시스턴트</h1><p>주택임대차보호법, 근로기준법, 개인정보보호법 관련 질문에 답변해 드립니다.</p>")
        chatbot_ui = gr.ChatInterface(
            fn=chatbot.chat,
            examples=example_questions,
            chatbot=gr.Chatbot(height=500, type='messages'),
            textbox=gr.Textbox(placeholder="질문을 입력하세요...", container=False, scale=7),
        )
    return demo

if __name__ == "__main__":
    app = create_chatbot_interface()
    app.launch()
