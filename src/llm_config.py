# ===================================================================
# [LLM 모듈: llm_config.py] (V3 - Colab 최종본 동기화)
# 역할: LLM 호출에 필요한 Pydantic 스키마, 프롬프트,
#      'get_fashion_advice_from_ai_analysis' 함수를 정의합니다.
#      (이 파일은 BE #2 (챗봇) 서버가 사용합니다.)
# ===================================================================

import os, json, re
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
from getpass import getpass

# --- 1. Pydantic 스키마 정의 ---
# (BE #1이 반환할 'analysis' JSON을 입력받을 스키마)
class AIAnalysis(BaseModel):
    context_pred: str
    main_material: str # (전체 평균)
    main_color: str    # (전체 평균)
    items: List[Dict[str, Any]] # (DetectedItem의 딕셔너리 리스트)
    # (H1, H2 점수 등도 필요시 추가)

# (LLM이 출력할 JSON 스키마)
class FashionAdvice(BaseModel):
    one_line_summary: str = Field(description="한 문장으로 된, 긍정적인 스타일 총평")
    positive_points: List[str] = Field(description="이 스타일의 매력적인 부분이나 칭찬할 만한 점 2-3가지")
    suggestion: str = Field(description="스타일을 한 단계 더 업그레이드할 수 있는 구체적이고 실천 가능한 아이템 또는 팁 1가지")

# --- 2. 시스템 프롬프트 정의 ---
SYSTEM_PROMPT = (
    """당신은 사용자의 자신감을 북돋우는 전문 AI 스타일리스트입니다. 
[목표]
- 입력으로 주어진 AI 분석 정보(맥락, 전체 색/재질, **개별 아이템 리스트**)를 바탕으로
  깔끔하고 실용적인 코디 피드백을 생성한다.
- 출력은 아래 JSON 템플릿의 **정확한 키**만 포함해야 한다.
[톤 & 문체]
- 한국어, 반말과 존댓말 사이의 중립적이고 공손한 서술체.
- 평가는 긍정 먼저 → 개선 제안 순서.
[아이템 목록(items) 활용 가이드]
- [FIX] 'items' 리스트에는 각 아이템의 이름('item'), **개별 색상('color')**, **개별 재질('material')**이 딕셔너리 형태로 들어있다.
- (예: `[{"item": "trousers", "color": "blue", "material": "denim"}, {"item": "short sleeve top", "color": "white", "material": "cotton"}]`)
- 'suggestion'을 작성할 때, 'items'에 있는 아이템 이름과 **"색상"**, **"재질"**을 
  **최소 1개 이상 구체적으로 언급**하여 조언한다.
- (예: "탐지된 'blue denim trousers'와 'white cotton top'의 색상/재질 조합이 좋습니다...")
- **[경고] 'main_color' (전체 색상)은 참고만 하고, 'items'의 개별 'color'/'material'을 항상 더 신뢰한다.**
[맥락별 가이드]
- consumer: 착용 편의/활동성/일상 TPO/재킷-하의 비율/신발 실용성/세탁 난이도 우선.
- shop: 시선집중 포인트/실루엣 대비/광택·매트 질감 대비/트렌드 키워드 강조.
[재질·색 조합 간단 룰셋]
- "blue" denim + "white" top: 클래식한 조합.
- "white" trousers + "blue" top: 시원한 마린룩.
- "beige" fur + "white" top: 부드럽고 따뜻한 톤온톤.
- "black" leather + "white" top: 시크한 대비.
[안전장치]
- 입력이 모호해도 보편 룰에 근거해 즉시 실행 가능한 제안을 낸다.
- 스키마 외 키 추가 금지, 줄바꿈/주석 금지."""
)

def _safe_parse_json(s: str) -> dict:
    """LLM이 JSON 앞뒤에 추가 텍스트를 붙여도 JSON만 파싱"""
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m: s = m.group(0)
    return json.loads(s)

# --- 3. [핵심] LLM 호출 함수 (BE #2가 사용) ---
# (이 함수는 "AI 분석 결과"를 입력받습니다)

def get_fashion_advice_from_ai_analysis(ai_analysis: AIAnalysis) -> FashionAdvice:
    """
    AI 분석 JSON(dict)을 입력받아, LLM을 호출하고, 
    'fashion_advice' JSON(dict)을 반환합니다.
    """
    
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if not client.api_key:
        raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    # [FIX] LLM에 전달할 (이름, 색상, 재질) 딕셔너리 리스트 생성
    items_for_llm = []
    for it in ai_analysis.items:
        name = it.get('h3_category') or it.get('category')
        
        color = "unknown"
        if it.get('colors_topk') and len(it['colors_topk']) > 0:
            color = it['colors_topk'][0].get('label', 'unknown')
        
        material = "unknown"
        if it.get('materials_topk') and len(it['materials_topk']) > 0:
            material = it['materials_topk'][0].get('label', 'unknown')
            
        items_for_llm.append({"item": name, "color": color, "material": material})
    
    # 중복 제거
    items_for_llm_unique = [dict(t) for t in {tuple(d.items()) for d in items_for_llm}]
    items_json_str = json.dumps(items_for_llm_unique, ensure_ascii=False) 
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"맥락: {ai_analysis.context_pred}\n"
            f"주요 재질 (전체): {ai_analysis.main_material}\n"
            f"주요 색 (전체): {ai_analysis.main_color}\n"
            f"탐지된 아이템 (이름, 개별 색상, 개별 재질): {items_json_str}\n"
            "아래 템플릿과 동일한 키만 포함한 JSON으로만 답하세요:\n"
            '{"one_line_summary":"","positive_points":[],"suggestion":""}'}
    ]
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7,
        response_format={"type": "json_object"},
    )
    content = completion.choices[0].message.content
    data = _safe_parse_json(content)
    return FashionAdvice.model_validate(data)

# --- 4. (테스트용) 이 파일이 직접 실행될 때만 작동 ---
if __name__ == "__main__":
    print("\n--- [LLM 모듈] llm_config.py 테스트 실행 ---")
    
    # 1. (가짜) AI 분석 결과 생성
    mock_ai_analysis_data = {
        "context_pred": "shop",
        "main_material": "denim", # (LLM이 무시해야 함)
        "main_color": "white",  # (LLM이 무시해야 함)
        "items": [
            {
                "category": "trousers",
                "h3_category": "trousers",
                "colors_topk": [{"label": "blue", "prob": 1.0}],
                "materials_topk": [{"label": "denim", "prob": 1.0}]
            },
            {
                "category": "short sleeve top",
                "h3_category": "short sleeve top",
                "colors_topk": [{"label": "white", "prob": 1.0}],
                "materials_topk": [{"label": "cotton", "prob": 1.0}]
            }
        ]
    }
    mock_ai_analysis = AIAnalysis.model_validate(mock_ai_analysis_data)
    
    # 2. OpenAI API 키 입력받기
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass("OpenAI API Key: ")

    try:
        # 3. LLM 조언 함수 실행
        fashion_advice = get_fashion_advice_from_ai_analysis(mock_ai_analysis)
        
        # 4. 결과 출력
        print("\n--- ✨ 최종 LLM 조언 JSON ✨ ---")
        print(fashion_advice.model_dump_json(indent=2, ensure_ascii=False))
        
    except Exception as e:
        print(f"\n[Error] LLM 테스트 실패: {e}")
