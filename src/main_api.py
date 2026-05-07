# ===================================================================
# [백엔드 #1: main_api.py]
# 역할: FastAPI를 사용하여 '/v1/analyze' API 엔드포인트를 생성합니다.
#      'fashion_model.py'를 임포트하여 AI 분석을 수행합니다.
# 실행: (터미널) uvicorn main_api:app --host 0.0.0.0 --port 8000
# ===================================================================

import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.responses import JSONResponse
import uvicorn

# --- 1. [핵심] AI 모듈 임포트 ---
# (이 import가 실행되는 순간, fashion_model.py가 실행되며
#  모든 AI 모델(YOLO, H1-H4)이 메모리에 로드됩니다.)
try:
    print("--- [BE #1] AI 모델 모듈(fashion_model.py) 로드 중... ---")
    import fashion_model
    print("--- [BE #1] AI 모델 모듈 로드 완료. ---")
except ImportError as e:
    print(f"[Fatal Error] 'fashion_model.py' 임포트 실패: {e}")
    print("    -> 'fashion_model.py' 파일이 같은 폴더에 있는지 확인하세요.")
    raise e
except FileNotFoundError as e:
    print(f"[Fatal Error] 모델 가중치 파일을 찾을 수 없습니다: {e}")
    print("    -> 'checkpoints' 및 'checkpoints_mixed' 폴더가 있는지 확인하세요.")
    raise e


# --- 2. FastAPI 앱 초기화 ---
app = FastAPI(
    title="Fashion AI Analysis API (BE #1)",
    description="이미지를 받아 AI 분석 JSON을 반환합니다.",
    version="1.0.0"
)

# 임시 업로드 및 작업 폴더
TEMP_UPLOAD_DIR = Path("./_temp_uploads")
E2E_RUN_DIR = Path("./_e2e_runs")
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
E2E_RUN_DIR.mkdir(exist_ok=True)


# --- 3. 헬스 체크 엔드포인트 ---
@app.get("/healthz", tags=["Status"])
def get_health_status():
    """서버가 살아있는지 확인합니다."""
    # (실제로는 AI 모델이 로드되었는지 등도 확인해야 함)
    return {"status": "ok", "message": "AI Model Server (BE #1) is running."}

# --- 4. [핵심] AI 분석 엔드포인트 ---
@app.post("/v1/analyze", tags=["Analysis"])
async def analyze_image(
    file: UploadFile = File(..., description="분석할 이미지 파일")
) -> Dict[str, Any]:
    """
    이미지 파일을 업로드받아 AI 모델로 분석하고,
    'analysis' JSON 객체를 반환합니다. (LLM 조언 제외)
    """
    
    # 1. 임시 파일로 저장
    # (이미지 파일명을 고유하게 만들어 동시성 문제 방지)
    safe_filename = f"upload_{os.urandom(8).hex()}{Path(file.filename).suffix}"
    temp_image_path = TEMP_UPLOAD_DIR / safe_filename
    
    try:
        with temp_image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"--- [BE #1] 이미지 수신: {file.filename} -> {temp_image_path} ---")

    except Exception as e:
        print(f"[Error] 파일 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    finally:
        file.file.close() # 파일 핸들 닫기

    # 2. [실행] AI 모듈(fashion_model.py)의 핵심 함수 호출
    try:
        # (str()로 경로를 문자열로 변환)
        ai_analysis_result = fashion_model.run_ai_analysis(
            image_path=str(temp_image_path),
            workdir=str(E2E_RUN_DIR)
        )
        
        if "error" in ai_analysis_result:
            raise HTTPException(status_code=500, detail=ai_analysis_result["error"])

        # 3. 성공적인 AI 분석 결과 반환
        print(f"--- [BE #1] 분석 완료. {len(ai_analysis_result['analysis']['items'])}개 아이템 탐지. ---")
        return ai_analysis_result # (inputs, analysis) 딕셔너리 반환

    except FileNotFoundError as e:
        print(f"[Error] 모델 가중치 파일 누락: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: Missing model weights. {e}")
    except Exception as e:
        print(f"[Error] AI 분석 중 오류: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {e}")
    finally:
        # (분석이 끝나면 임시 업로드 파일 삭제 - 선택 사항)
        if temp_image_path.exists():
            temp_image_path.unlink()

# --- 5. (테스트용) 서버 직접 실행 ---
if __name__ == "__main__":
    print("--- [BE #1] FastAPI 서버를 http://127.0.0.1:8000 에서 시작합니다. ---")
    print("--- (테스트) http://127.0.0.1:8000/docs 로 접속하여 'Try it out'을 사용하세요. ---")
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

---

**[실행 후 확인]**
이 `main_api.py` 파일은 당장 AWS 승인을 기다리는 동안, **로컬(WSL)에서 먼저 테스트**해볼 수 있습니다.

`bash` 터미널(`(deeplearning) epistachio@...`)에서 다음을 실행하세요:
1.  `(deeplearning)` 환경에 `fastapi`와 `uvicorn` 설치:
    ```bash
    pip install fastapi "uvicorn[standard]"
    ```
2.  FastAPI 서버 실행:
    ```bash
    python main_api.py