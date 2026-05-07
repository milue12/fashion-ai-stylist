# Fashion AI Stylist

## 프로젝트 소개

Fashion AI Stylist는 사용자의 전신 착장 이미지를 분석하여 패션 점수, 스타일 피드백, 추천 코디를 제공하는 AI 기반 패션 스타일리스트 서비스입니다.

본 프로젝트는 캡스톤 디자인 프로젝트로 진행되었으며, 사용자가 자신의 코디를 객관적으로 확인하고 개선할 수 있도록 돕는 것을 목표로 합니다.

## 주요 기능

- 전신 착장 이미지 업로드
- 착장 이미지 기반 패션 분석
- 패션 점수 제공
- 스타일 피드백 제공
- 추천 코디 및 개선 방향 제안

## 프로젝트 목적

사용자는 자신의 패션이 어울리는지 판단하기 어려운 경우가 많습니다.  
본 프로젝트는 AI를 활용하여 사용자의 착장을 분석하고, 점수와 피드백을 제공함으로써 더 나은 스타일 선택을 도와주는 것을 목표로 합니다.

## 기술 스택

- Python
- AI / Machine Learning
- Image Analysis

## 프로젝트 구조

```text
fashion-ai-stylist/
├── README.md
├── .gitignore
├── requirements.txt
├── notebooks/
│   └── Final_model.ipynb
├── src/
│   ├── chat_api.py
│   ├── fashion_model.py
│   ├── llm_config.py
│   ├── main_api.py
│   ├── models.py
│   └── utils.py
└── docs/
````

## 실행 방법

본 프로젝트의 주요 실험 및 모델 실행 코드는 `notebooks` 폴더 안의 Jupyter Notebook 파일에서 확인할 수 있습니다.

### 주요 노트북

- [Final_model.ipynb](notebooks/Final_model.ipynb)

### 실행 환경

본 프로젝트는 로컬 환경의 Jupyter Notebook을 기준으로 실행되었습니다.

- Python
- Jupyter Notebook
- PyTorch
- OpenCV
- YOLO

### 라이브러리 설치

프로젝트 실행에 필요한 라이브러리는 `requirements.txt` 파일에 정리되어 있습니다.

터미널 또는 Anaconda Prompt에서 다음 명령어를 실행합니다.


```bash
pip install -r requirements.txt
````

### 실행 순서

1. 저장소를 로컬 환경에 다운로드합니다.
2. 필요한 라이브러리를 설치합니다.
3. Jupyter Notebook을 실행합니다.
4. `notebooks/Final_model.ipynb` 파일을 엽니다.
5. 노트북 셀을 순서대로 실행합니다.
6. 이미지 입력 후 패션 분석 결과를 확인합니다.

## 주요 파일 설명

### notebooks

- `Final_model.ipynb`  
  최종 모델 노트북입니다.

### src

- `chat_api.py`  
  사용자 입력 또는 채팅 기반 응답 처리를 위한 코드입니다.

- `fashion_model.py`  
  패션 이미지 분석 모델 관련 코드입니다.

- `llm_config.py`  
  LLM 설정 관련 코드입니다.

- `main_api.py`  
  API 실행 및 연결을 위한 코드입니다.

- `models.py`  
  모델 구조 또는 데이터 처리에 필요한 클래스 정의 코드입니다.

- `utils.py`  
  프로젝트 전반에서 사용하는 보조 함수 코드입니다.
