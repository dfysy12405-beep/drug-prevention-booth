# 마약 2컷: 약물이 바꿔버린 미래

마약류 및 약물 오남용 예방 캠페인 체험 콘텐츠용 AI 필터 포토부스  
Python + Streamlit + OpenCV + MediaPipe 기반

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 주요 기능

- MediaPipe FaceMesh 기반 얼굴 감지 및 품질 검사
- 10가지 AFTER 필터 효과 (피부톤, 다크서클, 충혈, 치아 손상 등)
- 머그샷 배경 오버레이
- 관리자 설정 (필터 강도 조절, 프리셋)
- 개인정보 보호 (이미지 미저장)
- PNG 다운로드

## 배포

### Streamlit Community Cloud
1. GitHub에 코드 업로드
2. share.streamlit.io 에서 연결
3. Main file: app.py

### 관리자 비밀번호
기본값: `1234` (app.py 내 ADMIN_PW 변수 수정)
