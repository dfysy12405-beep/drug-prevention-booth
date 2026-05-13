# 마약 2컷: 약물이 바꿔버린 미래 (Streamlit 버전)

마약류 및 약물 오남용 예방 캠페인 체험 콘텐츠용 AI 필터 포토부스입니다.
Python + Streamlit + OpenCV + MediaPipe 기반으로 제작되었습니다.

## 개요

참가자가 태블릿 PC 또는 모바일 기기 카메라로 얼굴을 촬영하면, "약물이 바꿔버린 미래 모습" 컨셉의 BEFORE / AFTER 이미지가 생성됩니다.

## 주요 기능

- **얼굴 감지**: MediaPipe FaceMesh 기반 468개 랜드마크 감지
- **품질 검사**: 얼굴 위치, 크기, 정면 여부, 밝기 자동 검사
- **AFTER 필터**: 피부톤 회색화, 다크서클, 눈 충혈, 피부 거칠기, 입 주변 착색, 볼 꺼짐, 치아 손상, 머그샷 배경 등 10가지 효과
- **관리자 설정**: 필터 강도 조절, 프리셋(약하게/보통/강하게)
- **개인정보 보호**: 촬영 이미지 미저장, 세션 메모리에서만 처리
- **다운로드**: PNG 파일 다운로드 지원

## 파일 구조

```
├── app.py              # 메인 Streamlit 앱
├── requirements.txt    # Python 의존성
├── assets/             # 에셋 폴더 (확장용)
└── README.md           # 이 파일
```

## 로컬 실행

### 1. 환경 설정

```bash
# Python 3.9+ 필요
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 자동 접속됩니다.

### 3. 태블릿 테스트

```bash
# 네트워크 접근 허용
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

같은 네트워크의 태블릿에서 `http://<PC_IP>:8501` 접속

## 배포

### Streamlit Community Cloud

1. GitHub에 코드 업로드
2. [share.streamlit.io](https://share.streamlit.io) 접속
3. "New app" → 리포지토리 선택
4. Main file: `app.py`
5. Deploy 클릭

> **참고**: `opencv-python-headless`를 사용하므로 시스템 패키지 설치 불필요

### Render

1. GitHub 연동
2. New Web Service 생성
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port $PORT --server.headless true`
5. Deploy

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
```

```bash
docker build -t photobooth .
docker run -p 8501:8501 photobooth
```

## 관리자 설정

- 시작 화면 하단 ⚙️ 버튼 클릭
- 기본 비밀번호: `1234` (app.py에서 `ADMIN_PASSWORD` 변경 가능)
- 필터 강도 슬라이더 10개 + 프리셋 3종
- 머그샷 배경 ON/OFF, 디버그 모드 ON/OFF

## 개인정보 보호

- 촬영 이미지는 **서버/DB에 저장하지 않습니다**
- Streamlit `session_state` 메모리에서만 임시 처리
- 세션 종료/새로고침 시 자동 삭제
- 다운로드는 사용자 직접 선택 시에만 가능

## 기술 스택

| 구성 | 기술 |
|------|------|
| 프레임워크 | Streamlit |
| 얼굴 감지 | MediaPipe FaceMesh |
| 이미지 처리 | OpenCV, NumPy, Pillow |
| 언어 | Python 3.9+ |

## 커스터마이즈

### 비밀번호 변경
```python
ADMIN_PASSWORD = "새비밀번호"
```

### 기본 필터 강도 변경
```python
DEFAULT_SETTINGS = {
    "skinGray": 60,    # 0~100
    "darkCircle": 55,
    ...
}
```

### 새 프리셋 추가
```python
PRESETS["커스텀"] = {
    "skinGray": 70,
    "darkCircle": 65,
    ...
}
```

## 라이선스

본 프로젝트는 마약류 오남용 예방 캠페인 목적으로 제작되었습니다.
