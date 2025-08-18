# LangGraph_Chatbot
- AI 에이전트로 구현하는 RAG 시스템 강의를 바탕으로 제작한 **LangGraph 기반 챗봇**입니다.

## 프로젝트 구조
```
langgraph_chatbot/
├── chroma_db/             # ChromaDB 벡터 저장소
├── .env                   # API 키 등 환경 변수
├── pyproject.toml         # Python 의존성 관리
├── README.md              # README
└── src/
├── app.py             # Gradio 실행
├── agents.py          # RAG 에이전트 로직
├── graph.py           # 메인 LangGraph 로직
├── tools.py           # 검색 도구 정의
├── config.py          # 설정
└── logger_config.py   # 로깅 설정
```
## 기술 스택
<div align=left> 
  <img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white"> 
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=LangChain&logoColor=white">
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=LangGraph&logoColor=white">
  <img src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=OpenAI&logoColor=white">
  <img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=HuggingFace&logoColor=white">
  <img src="https://img.shields.io/badge/Pydantic-E92063?style=for-the-badge&logo=Pydantic&logoColor=white"> 
  <img src="https://img.shields.io/badge/Gradio-F97316?style=for-the-badge&logo=Gradio&logoColor=white">
  <img src="https://img.shields.io/badge/ChromaDB-6665CD?style=for-the-badge">

</div>

## 설치 및 실행
### 1. 가상 환경 생성 및 활성화
- 이 프로젝트는 `uv`를 사용하여 가상 환경 및 패키지를 관리합니다.

**`uv` 설치 방법**
  
- macOS 및 리눅스
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
- Windows
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
- pip를 통한 설치
```
pip install uv
```

**가상 환경 생성**
```bash
uv venv
```

**가상 환경 활성화(Windows)**
```
.venv\Scripts\activate
```

**가상 환경 활성화(macOS/Linux)**
```
source .venv/bin/activate
```

### 2. 의존성 패키지 설치
```
uv sync
```

### 3. 환경 변수 설정
- .env 파일 생성
- .env 파일 내용
```
OPENAI_API_KEY="sk-..."
TAVILY_API_KEY="tvly-..."
```

### 4. 실행
```
cd src
uv run python app.py
```
- 실행 후 나타나는 로컬 url(http://127.0.0.1:7860) 을 웹 브라우저에서 열어 챗봇을 사용

### 5. 챗봇 실행 결과
<img width="1898" height="968" alt="Image" src="https://github.com/user-attachments/assets/b7e504ca-2f72-457b-b041-e7e294a310fd" />

## LangGraph 구조
<img width="696" height="432" alt="Image" src="https://github.com/user-attachments/assets/7c7f473a-97b4-4729-b8b0-8e9b552a85cf" />
