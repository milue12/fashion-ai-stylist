# ===================================================================
# [백엔드 #2: chat_api.py]
# 역할: FastAPI를 사용하여 '/v1/chat' API 엔드포인트를 생성합니다.
#      'llm_config.py'를 임포트하여 LLM 조언을 수행합니다.
# 실행: (터미널) uvicorn chat_api:app --host 0.0.0.0 --port 8001
# ===================================================================

import os  # <-- [수정됨] 'import os'가 맨 위로 왔습니다.
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
import uvicorn
import httpx # [NEW] BE #1을 호출하기 위한 비동기 HTTP 클라이언트
from typing import Dict, Any

# --- 1. [핵심] LLM 모듈 임포트 ---
# (이 import는 AI 모델을 로드하지 않으므로, GPU 없이도 빠릅니다)
try:
    print("--- [BE #2] LLM 설정 모듈(llm_config.py) 로드 중... ---")
    # AIAnalysis 스키마와 LLM 호출 함수를 가져옵니다.
    from llm_config import AIAnalysis, FashionAdvice, get_fashion_advice_from_ai_analysis
    print("--- [BE #2] LLM 설정 모듈 로드 완료. ---")
except ImportError as e:
    print(f"[Fatal Error] 'llm_config.py' 임포트 실패: {e}")
    print("    -> 'llm_config.py' 파일이 같은 폴더에 있는지 확인하세요.")
    raise e

# --- 2. FastAPI 앱 초기화 ---
app = FastAPI(
    title="Fashion Chatbot API (BE #2)",
    description="이미지를 받아 AI 분석을 요청하고, LLM 조언을 반환합니다.",
    version="1.0.0"
)

# [중요] BE #1 (AI 모델 서버)의 주소.
# (AWS 승인 후 EC2 비공개 IP로 변경됩니다.)
AI_API_ENDPOINT = os.environ.get("A_API_ENDPOINT", "http://54.173.95.105:8000/v1/analyze") # <-- [수정됨] AWS IP 주소
print(f"--- [BE #2] AI 서버(BE #1) 타겟 주소: {AI_API_ENDPOINT} ---")

# --- 3. 헬스 체크 엔드포인트 ---
@app.get("/healthz", tags=["Status"])
def get_health_status():
    """서버가 살아있는지 확인합니다."""
    return {"status": "ok", "message": "Chatbot Server (BE #2) is running."}

# --- 4. [핵심] 챗봇 엔드포인트 ---
@app.post("/v1/chat", tags=["Chat"])
async def handle_chat_with_image(
    file: UploadFile = File(..., description="분석할 이미지 파일"),
    # (추후 세션 ID 등도 받을 수 있음)
    # session_id: str = Body(None, description="채팅 세션 ID")
) -> Dict[str, Any]:
    """
    (MVP) 이미지 1장을 받아, BE #1에 AI 분석을 요청하고,
    그 결과를 LLM에 전달하여 "패션 조언" JSON을 반환합니다.
    """
    
    print(f"--- [BE #2] /v1/chat 요청 수신: {file.filename} ---")
    
    # --- A. BE #1 (AI 모델 서버) 호출 ---
    print(f"--- [BE #2] BE #1 ({AI_API_ENDPOINT}) 호출 중... ---")
    
    # 업로드된 파일 데이터를 'files' 딕셔너리로 준비
    files_to_upload = {'file': (file.filename, file.file, file.content_type)}
    
    ai_analysis_result = None
    try:
        # httpx를 사용하여 BE #1에 비동기 POST 요청 (S2S 통신)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(AI_API_ENDPOINT, files=files_to_upload)
            
            if response.status_code == 200:
                ai_analysis_result = response.json()
                print("--- [BE #2] BE #1로부터 AI 분석 JSON 수신 완료. ---")
            else:
                print(f"[Error] BE #1 호출 실패 (Status: {response.status_code}): {response.text}")
                raise HTTPException(status_code=502, detail=f"AI Analysis Server (BE #1) FAILED: {response.text}")

    except httpx.ConnectError as e:
        print(f"[Fatal Error] BE #1 ({AI_API_ENDPOINT})에 연결할 수 없습니다.")
        print("    -> BE #1 (main_api.py)이 'AWS EC2'에서 실행 중인지 확인하세요.")
        raise HTTPException(status_code=503, detail=f"AI Analysis Server (BE #1) is unreachable: {e}")
    except Exception as e:
        print(f"[Error] BE #1 호출 중 알 수 없는 오류: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error during AI analysis call: {e}")

    # --- B. LLM (OpenAI) 호출 ---
    if not ai_analysis_result or "analysis" not in ai_analysis_result:
        raise HTTPException(status_code=500, detail="Invalid response from AI Server (BE #1).")

    try:
        # 1. BE #1의 결과에서 'analysis' 부분만 추출
        ai_analysis_data = ai_analysis_result["analysis"]
        
        # 2. Pydantic 스키마로 검증
        ai_analysis_obj = AIAnalysis.model_validate(ai_analysis_data)
        
        print(f"--- [BE #2] LLM (gpt-4o-mini) 호출 중... ---")
        
        # 3. 'llm_config.py'의 함수를 호출하여 조언 생성
        fashion_advice = get_fashion_advice_from_ai_analysis(ai_analysis_obj)
        
        # 4. 최종 결과 반환
        print(f"--- [BE #2] LLM 조언 생성 완료. 200 OK 응답. ---")
        return {
            "ai_analysis": ai_analysis_data, # (디버깅용) AI 분석 결과 포함
            "fashion_advice": fashion_advice.model_dump() # LLM 조언 결과
        }

    except Exception as e:
        print(f"[Error] LLM 조언 생성 중 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"LLM advice generation failed: {e}")

# --- 5. (테스트용) 서버 직접 실행 ---
if __name__ == "__main__":
    print("--- [BE #2] FastAPI 서버를 http://127.0.0.1:8001 에서 시작합니다. ---")
    print(f"--- [BE #2] AI 서버(BE #1) 타겟: {AI_API_ENDPOINT} ---")
    print("--- (테스트) http://127.0.0.1:8001/docs 로 접속하여 'Try it out'을 사용하세요. ---")
    uvicorn.run(app, host="127.0.0.1", port=8001)
