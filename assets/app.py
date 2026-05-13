"""
마약 2컷: 약물이 바꿔버린 미래
=============================
마약류 및 약물 오남용 예방 캠페인용 AI 필터 포토부스
Python + Streamlit + OpenCV + MediaPipe 기반

실행: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import json
import time
from datetime import datetime

# ============================================
# 페이지 설정
# ============================================
st.set_page_config(
    page_title="마약 2컷: 약물이 바꿔버린 미래",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================
# CSS 스타일
# ============================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&family=Black+Han+Sans&display=swap');

/* 전역 다크 테마 */
.stApp {
    background-color: #0a0a0f !important;
    color: #f1f1f4 !important;
}

/* 사이드바 숨김 */
[data-testid="stSidebar"] { display: none; }
header[data-testid="stHeader"] { background: transparent !important; }

/* 시작화면 */
.start-container {
    text-align: center;
    padding: 2rem 1rem;
    max-width: 540px;
    margin: 0 auto;
    animation: fadeIn 0.8s ease-out;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.start-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    background: rgba(230,57,70,0.15);
    border: 1px solid rgba(230,57,70,0.3);
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 500;
    color: #e63946;
    letter-spacing: 0.05em;
    margin-bottom: 1.5rem;
}
.start-title {
    font-family: 'Black Han Sans', sans-serif;
    font-size: clamp(2.5rem, 8vw, 4.5rem);
    color: #f1f1f4;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.start-title .accent { color: #e63946; }
.start-subtitle {
    font-family: 'Black Han Sans', sans-serif;
    font-size: clamp(1rem, 3.5vw, 1.6rem);
    color: #8a8a9e;
    margin-bottom: 2rem;
}
.start-desc {
    font-size: 0.95rem;
    color: #8a8a9e;
    line-height: 1.7;
    margin-bottom: 2rem;
}
.start-desc strong { color: #f1f1f4; }
.privacy-notice {
    font-size: 0.75rem;
    color: #55556a;
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* 결과화면 */
.result-header {
    text-align: center;
    font-family: 'Black Han Sans', sans-serif;
    font-size: clamp(1.2rem, 4vw, 1.8rem);
    color: #f1f1f4;
    margin-bottom: 0.5rem;
}
.result-message {
    text-align: center;
    font-size: 0.85rem;
    color: #8a8a9e;
    margin-bottom: 1rem;
}
.label-before {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    background: #3498db;
    color: white;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.label-after {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    background: #e63946;
    color: white;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

/* 안내 메시지 */
.guide-msg {
    text-align: center;
    padding: 0.75rem;
    border-radius: 8px;
    font-weight: 500;
    margin: 0.5rem 0;
}
.guide-msg.success {
    background: rgba(46,204,113,0.1);
    border: 1px solid rgba(46,204,113,0.3);
    color: #2ecc71;
}
.guide-msg.warning {
    background: rgba(243,156,18,0.1);
    border: 1px solid rgba(243,156,18,0.3);
    color: #f39c12;
}
.guide-msg.error {
    background: rgba(231,76,60,0.1);
    border: 1px solid rgba(231,76,60,0.3);
    color: #e74c3c;
}

/* 관리자 설정 */
.admin-header {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: #f1f1f4;
}

/* 버튼 스타일링 */
.stButton > button {
    font-family: 'Noto Sans KR', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    transition: all 0.2s !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button {
    width: 100%;
}

/* 카메라 입력 스타일 */
[data-testid="stCameraInput"] {
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stCameraInput"] video {
    transform: scaleX(-1);
}

/* 디버그 정보 */
.debug-box {
    background: rgba(0,0,0,0.6);
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 0.5rem 0.75rem;
    font-family: monospace;
    font-size: 0.75rem;
    color: #2ecc71;
}
</style>
"""

# ============================================
# 상수 정의
# ============================================

# MediaPipe FaceMesh 주요 랜드마크 인덱스
class LM:
    """얼굴 랜드마크 인덱스 상수"""
    NOSE_TIP = 1
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14
    LIP_TOP = 0
    LIP_BOTTOM = 17
    LEFT_CHEEK = 234
    RIGHT_CHEEK = 454
    CHIN = 152
    FOREHEAD = 10
    LEFT_TEMPLE = 127
    RIGHT_TEMPLE = 356

    # 얼굴 윤곽
    FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

    # 눈 아래 (다크서클)
    LEFT_UNDER_EYE = [111, 117, 118, 119, 120, 121, 128]
    RIGHT_UNDER_EYE = [340, 346, 347, 348, 349, 350, 357]

    # 입술 외곽
    LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                  291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

    # 왼쪽/오른쪽 눈 외곽
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]


# 기본 설정값
DEFAULT_SETTINGS = {
    "skinGray": 60,
    "darkCircle": 55,
    "eyeRedness": 45,
    "dullEyes": 40,
    "skinRough": 50,
    "mouthStain": 50,
    "cheekHollow": 45,
    "mouthDroop": 35,
    "asymmetry": 25,
    "teethDamage": 50,
    "mugshot": True,
    "autoCapture": False,
    "debug": False,
}

PRESETS = {
    "약하게": {"skinGray": 35, "darkCircle": 30, "eyeRedness": 25, "dullEyes": 20,
              "skinRough": 25, "mouthStain": 25, "cheekHollow": 20, "mouthDroop": 15,
              "asymmetry": 10, "teethDamage": 25},
    "보통": {"skinGray": 60, "darkCircle": 55, "eyeRedness": 45, "dullEyes": 40,
            "skinRough": 50, "mouthStain": 50, "cheekHollow": 45, "mouthDroop": 35,
            "asymmetry": 25, "teethDamage": 50},
    "강하게": {"skinGray": 85, "darkCircle": 80, "eyeRedness": 70, "dullEyes": 65,
              "skinRough": 75, "mouthStain": 75, "cheekHollow": 70, "mouthDroop": 55,
              "asymmetry": 40, "teethDamage": 75},
}

ADMIN_PASSWORD = "1234"


# ============================================
# 세션 상태 초기화
# ============================================
def init_session_state():
    """세션 상태 초기화 - 이미지는 메모리에서만 관리"""
    defaults = {
        "screen": "start",        # start, camera, result, admin
        "settings": {**DEFAULT_SETTINGS},
        "before_image": None,     # numpy array (메모리 전용)
        "after_image": None,      # numpy array (메모리 전용)
        "composite_image": None,  # numpy array (메모리 전용)
        "landmarks": None,        # 감지된 랜드마크
        "quality_info": None,     # 품질 검사 결과
        "admin_auth": False,      # 관리자 인증 여부
        "camera_key": 0,          # 카메라 리셋용 키
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # localStorage에서 설정값만 복원 (이미지 절대 저장 안함)
    # Streamlit에서는 session_state 사용


def clear_images():
    """개인정보 보호: 모든 이미지 즉시 삭제"""
    st.session_state.before_image = None
    st.session_state.after_image = None
    st.session_state.composite_image = None
    st.session_state.landmarks = None
    st.session_state.quality_info = None


def go_to_screen(screen_name: str):
    """화면 전환"""
    if screen_name in ("start", "camera"):
        clear_images()
    st.session_state.screen = screen_name
    if screen_name == "camera":
        st.session_state.camera_key += 1


# ============================================
# MediaPipe FaceMesh 초기화
# ============================================
@st.cache_resource
def get_face_mesh():
    """MediaPipe FaceMesh 싱글톤"""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )


# ============================================
# 얼굴 감지 및 품질 검사
# ============================================
def detect_face(image: np.ndarray):
    """
    얼굴 감지 및 랜드마크 추출

    Returns:
        (landmarks, quality_info) 또는 (None, quality_info)
    """
    face_mesh = get_face_mesh()
    h, w = image.shape[:2]

    # BGR → RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    quality = {
        "detected": False,
        "center_ok": False,
        "size_ok": False,
        "frontal_ok": False,
        "brightness_ok": False,
        "all_passed": False,
        "message": "",
        "face_size": 0,
        "yaw": 0,
        "brightness": 0,
    }

    if not results.multi_face_landmarks:
        quality["message"] = "얼굴이 감지되지 않았습니다. 카메라를 바라봐주세요."
        return None, quality

    if len(results.multi_face_landmarks) > 1:
        quality["message"] = "한 명만 카메라 앞에 서주세요."
        return None, quality

    landmarks = results.multi_face_landmarks[0].landmark
    quality["detected"] = True

    # 얼굴 중심 (코 끝)
    nose = landmarks[LM.NOSE_TIP]
    cx, cy = nose.x, nose.y

    # 중앙 체크
    quality["center_ok"] = 0.25 < cx < 0.75 and 0.15 < cy < 0.75

    # 크기 체크 (눈 간 거리)
    l_eye = landmarks[LM.LEFT_EYE_OUTER]
    r_eye = landmarks[LM.RIGHT_EYE_OUTER]
    eye_dist = np.hypot(l_eye.x - r_eye.x, l_eye.y - r_eye.y)
    quality["face_size"] = eye_dist

    too_small = eye_dist < 0.12
    too_large = eye_dist > 0.45
    quality["size_ok"] = not too_small and not too_large

    # 정면 체크 (Yaw 추정)
    nose_bridge = landmarks[6]
    left_temple = landmarks[LM.LEFT_TEMPLE]
    right_temple = landmarks[LM.RIGHT_TEMPLE]
    face_w = abs(right_temple.x - left_temple.x)
    if face_w > 0:
        nose_offset = (nose_bridge.x - left_temple.x) / face_w
        yaw = (nose_offset - 0.5) * 90
    else:
        yaw = 0
    quality["yaw"] = yaw
    quality["frontal_ok"] = abs(yaw) < 18

    # 밝기 체크
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    quality["brightness"] = brightness
    quality["brightness_ok"] = brightness > 50

    # 종합 판정
    quality["all_passed"] = all([
        quality["center_ok"],
        quality["size_ok"],
        quality["frontal_ok"],
        quality["brightness_ok"],
    ])

    # 메시지
    if not quality["center_ok"]:
        quality["message"] = "얼굴을 화면 중앙에 맞춰주세요."
    elif too_small:
        quality["message"] = "얼굴이 너무 작습니다. 조금 가까이 와주세요."
    elif too_large:
        quality["message"] = "얼굴이 너무 큽니다. 조금 뒤로 이동해주세요."
    elif not quality["frontal_ok"]:
        quality["message"] = "정면을 바라봐주세요."
    elif not quality["brightness_ok"]:
        quality["message"] = "조명이 어두워 얼굴 인식이 어렵습니다."
    else:
        quality["message"] = "좋습니다! 얼굴 인식이 완료되었습니다."

    return landmarks, quality


# ============================================
# 얼굴 마스크 생성
# ============================================
def create_face_mask(landmarks, w: int, h: int) -> np.ndarray:
    """얼굴 윤곽 기반 마스크 생성 (0~255)"""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = []
    for idx in LM.FACE_OVAL:
        lm = landmarks[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    # 부드러운 경계
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    return mask


def get_landmark_point(landmarks, idx, w, h):
    """랜드마크 좌표를 픽셀 좌표로 변환"""
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def get_landmark_points(landmarks, indices, w, h):
    """여러 랜드마크의 중심점"""
    xs, ys = [], []
    for idx in indices:
        lm = landmarks[idx]
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    return int(np.mean(xs)), int(np.mean(ys))


def get_eye_distance(landmarks, w, h):
    """눈 간 거리 (픽셀)"""
    lx, ly = get_landmark_point(landmarks, LM.LEFT_EYE_OUTER, w, h)
    rx, ry = get_landmark_point(landmarks, LM.RIGHT_EYE_OUTER, w, h)
    return np.hypot(lx - rx, ly - ry)


# ============================================
# AFTER 필터 효과
# ============================================

def apply_skin_gray(image: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
    """피부톤 회색화 - 채도 감소 + 누런/회색 톤"""
    result = image.copy()
    h, w = image.shape[:2]

    # HSV로 변환하여 채도 감소
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    # 채도 감소
    hsv[:, :, 1] *= (1 - intensity * 0.7)
    # 약간 노란 쪽으로 Hue 이동
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + intensity * 5, 0, 180)
    # 명도 약간 감소
    hsv[:, :, 2] *= (1 - intensity * 0.1)
    hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
    desaturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 마스크 기반 블렌딩
    mask_f = (mask / 255.0)[:, :, np.newaxis]
    result = (result * (1 - mask_f * intensity) + desaturated * mask_f * intensity).astype(np.uint8)
    return result


def apply_skin_roughness(image: np.ndarray, mask: np.ndarray, intensity: float) -> np.ndarray:
    """피부 거칠기 - 노이즈 + 대비 증가"""
    result = image.copy()
    h, w = image.shape[:2]

    # 노이즈 생성
    noise = np.random.randn(h, w, 3) * (intensity * 18)
    noise = noise.astype(np.float32)

    # 마스크 적용
    mask_f = (mask / 255.0)[:, :, np.newaxis]
    result = np.clip(result.astype(np.float32) + noise * mask_f, 0, 255).astype(np.uint8)

    # 약간의 대비 증가 (얼굴 영역만)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=1.0 + intensity * 1.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    # 대비 증가분만 블렌딩
    diff = enhanced_gray.astype(np.float32) - gray.astype(np.float32)
    for c in range(3):
        channel = result[:, :, c].astype(np.float32)
        channel += diff * mask_f[:, :, 0] * intensity * 0.3
        result[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    return result


def draw_dark_circles(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """다크서클 - 눈 밑 어두운 그림자"""
    result = image.copy()
    h, w = image.shape[:2]
    eye_dist = get_eye_distance(landmarks, w, h)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for eye_points in [LM.LEFT_UNDER_EYE, LM.RIGHT_UNDER_EYE]:
        cx, cy = get_landmark_points(landmarks, eye_points, w, h)
        rx = int(eye_dist * 0.18)
        ry = int(eye_dist * 0.08)
        cv2.ellipse(overlay, (cx, cy + ry // 2), (rx, ry), 0, 0, 360,
                     (30, 15, 40), -1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), eye_dist * 0.04)
    alpha = intensity * 0.5
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_eye_redness(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """눈 충혈 - 붉은 반투명 오버레이"""
    result = image.copy()
    h, w = image.shape[:2]
    eye_dist = get_eye_distance(landmarks, w, h)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for eye_indices in [LM.LEFT_EYE, LM.RIGHT_EYE]:
        cx, cy = get_landmark_points(landmarks, eye_indices, w, h)
        r = int(eye_dist * 0.1)
        # 붉은 충혈
        cv2.ellipse(overlay, (cx, cy), (int(r * 1.2), int(r * 0.7)),
                     0, 0, 360, (20, 30, 180), -1)
        # 약간의 노란기
        cv2.ellipse(overlay, (cx, cy), (int(r * 0.9), int(r * 0.5)),
                     0, 0, 360, (10, 100, 160), -1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), eye_dist * 0.03)
    alpha = intensity * 0.3
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_dull_eyes(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """멍한 눈 - 눈 주위 어두운 그림자"""
    result = image.copy()
    h, w = image.shape[:2]
    eye_dist = get_eye_distance(landmarks, w, h)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for eye_indices in [LM.LEFT_EYE, LM.RIGHT_EYE]:
        cx, cy = get_landmark_points(landmarks, eye_indices, w, h)
        r = int(eye_dist * 0.2)
        cv2.ellipse(overlay, (cx, cy - int(r * 0.3)),
                     (int(r * 1.3), r), 0, 0, 360,
                     (25, 10, 30), -1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), eye_dist * 0.06)
    alpha = intensity * 0.3
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_mouth_stain(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """입 주변 착색 - 적갈색 오버레이"""
    result = image.copy()
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)

    mlx, mly = get_landmark_point(landmarks, LM.MOUTH_LEFT, w, h)
    mrx, mry = get_landmark_point(landmarks, LM.MOUTH_RIGHT, w, h)
    mtx, mty = get_landmark_point(landmarks, LM.MOUTH_TOP, w, h)
    mbx, mby = get_landmark_point(landmarks, LM.MOUTH_BOTTOM, w, h)

    mcx = (mlx + mrx) // 2
    mcy = (mty + mby) // 2
    mw = abs(mrx - mlx)
    mh = abs(mby - mty)

    # 입 주변 넓은 어두운 착색
    cv2.ellipse(overlay, (mcx, mcy), (int(mw * 0.7), int(mh * 1.5)),
                 0, 0, 360, (15, 15, 60), -1)
    # 입술 영역 더 어둡게
    cv2.ellipse(overlay, (mcx, mcy), (int(mw * 0.5), int(mh * 0.8)),
                 0, 0, 360, (15, 10, 40), -1)

    blur_size = max(3, int(mw * 0.15)) | 1
    overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), mw * 0.08)
    alpha = intensity * 0.4
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_cheek_hollow(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """볼 꺼짐 - 광대 아래 그림자"""
    result = image.copy()
    h, w = image.shape[:2]
    eye_dist = get_eye_distance(landmarks, w, h)
    overlay = np.zeros_like(image, dtype=np.uint8)

    lcx, lcy = get_landmark_point(landmarks, LM.LEFT_CHEEK, w, h)
    rcx, rcy = get_landmark_point(landmarks, LM.RIGHT_CHEEK, w, h)
    mlx, mly = get_landmark_point(landmarks, LM.MOUTH_LEFT, w, h)
    mrx, mry = get_landmark_point(landmarks, LM.MOUTH_RIGHT, w, h)

    r = int(eye_dist * 0.18)

    # 왼쪽 볼 꺼짐
    cx1 = int(lcx * 0.6 + mlx * 0.4)
    cy1 = int(lcy * 0.4 + mly * 0.6)
    cv2.ellipse(overlay, (cx1, cy1), (int(r * 1.2), int(r * 0.7)),
                 -15, 0, 360, (20, 10, 25), -1)

    # 오른쪽 볼 꺼짐
    cx2 = int(rcx * 0.6 + mrx * 0.4)
    cy2 = int(rcy * 0.4 + mry * 0.6)
    cv2.ellipse(overlay, (cx2, cy2), (int(r * 1.2), int(r * 0.7)),
                 15, 0, 360, (20, 10, 25), -1)

    overlay = cv2.GaussianBlur(overlay, (0, 0), r * 0.7)
    alpha = intensity * 0.35
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_mouth_droop(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """입꼬리 하강 - 그림자 기반"""
    result = image.copy()
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)

    mlx, mly = get_landmark_point(landmarks, LM.MOUTH_LEFT, w, h)
    mrx, mry = get_landmark_point(landmarks, LM.MOUTH_RIGHT, w, h)
    mw = abs(mrx - mlx)
    r = int(mw * 0.06)

    # 입꼬리 아래 그림자
    cv2.ellipse(overlay, (mlx, mly + r * 2), (r * 2, r * 3),
                 0, 0, 360, (20, 10, 25), -1)
    cv2.ellipse(overlay, (mrx, mry + r * 2), (r * 2, r * 3),
                 0, 0, 360, (20, 10, 25), -1)

    blur_size = max(3, r * 3) | 1
    overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), r * 1.2)
    alpha = intensity * 0.4
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_asymmetry(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """약한 얼굴 비대칭 - 한쪽 어둡게"""
    result = image.copy()
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.uint8)

    nx, ny = get_landmark_point(landmarks, LM.NOSE_TIP, w, h)
    fx, fy = get_landmark_point(landmarks, LM.FOREHEAD, w, h)
    chx, chy = get_landmark_point(landmarks, LM.CHIN, w, h)

    cx = nx + int(w * 0.06)
    cy = (fy + chy) // 2
    rx = int(w * 0.15)
    ry = int(abs(chy - fy) * 0.5)

    cv2.ellipse(overlay, (cx, cy), (rx, ry), 0, 0, 360, (15, 8, 20), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), rx * 0.4)
    alpha = intensity * 0.15
    result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    return result


def draw_teeth_damage(image: np.ndarray, landmarks, intensity: float) -> np.ndarray:
    """치아 손상 - 어두운 틈 + 누런 변색"""
    result = image.copy()
    h, w = image.shape[:2]

    mlx, mly = get_landmark_point(landmarks, LM.MOUTH_LEFT, w, h)
    mrx, mry = get_landmark_point(landmarks, LM.MOUTH_RIGHT, w, h)
    mtx, mty = get_landmark_point(landmarks, LM.MOUTH_TOP, w, h)
    mbx, mby = get_landmark_point(landmarks, LM.MOUTH_BOTTOM, w, h)

    mcx = (mlx + mrx) // 2
    mcy = (mty + mby) // 2
    mw = abs(mrx - mlx)
    mh = abs(mby - mty)

    # 입 안쪽 어두운 영역
    overlay_dark = np.zeros_like(image, dtype=np.uint8)
    cv2.ellipse(overlay_dark, (mcx, mcy - int(mh * 0.15)),
                 (int(mw * 0.2), int(mh * 0.25)), 0, 0, 360,
                 (5, 5, 20), -1)
    overlay_dark = cv2.GaussianBlur(overlay_dark, (0, 0), mw * 0.02)
    result = cv2.addWeighted(result, 1.0, overlay_dark, intensity * 0.5, 0)

    # 누런 치아 (입술 틈 사이)
    tooth_y = mcy - int(mh * 0.15)
    tooth_w = int(mw * 0.06)
    tooth_h = int(mh * 0.2)

    np.random.seed(42)  # 일관된 결과
    for i in range(-2, 3):
        tx = mcx + i * int(tooth_w * 1.4)
        yellow_val = 0.4 + np.random.random() * 0.3

        # 누런 치아
        color = (int(30 + np.random.random() * 20),
                 int(100 + np.random.random() * 30),
                 int(140 + np.random.random() * 40))
        cv2.rectangle(result,
                       (tx - tooth_w // 2, tooth_y - tooth_h // 2),
                       (tx + tooth_w // 2, tooth_y + tooth_h // 2),
                       color, -1)

        # 일부 치아에 검은 틈
        if np.random.random() < intensity * 0.5:
            gap_w = max(1, int(mw * 0.006))
            cv2.rectangle(result,
                           (tx + tooth_w // 2 - gap_w, tooth_y - tooth_h // 2),
                           (tx + tooth_w // 2, tooth_y + tooth_h // 2),
                           (5, 5, 10), -1)

    # 블러로 자연스럽게
    region_y1 = max(0, tooth_y - tooth_h)
    region_y2 = min(h, tooth_y + tooth_h)
    region_x1 = max(0, mcx - int(mw * 0.25))
    region_x2 = min(w, mcx + int(mw * 0.25))
    if region_y2 > region_y1 and region_x2 > region_x1:
        region = result[region_y1:region_y2, region_x1:region_x2]
        blur_k = max(3, int(mw * 0.02)) | 1
        result[region_y1:region_y2, region_x1:region_x2] = cv2.GaussianBlur(
            region, (blur_k, blur_k), 0
        )

    return result


def apply_color_grading(image: np.ndarray) -> np.ndarray:
    """차가운 톤 색보정 + 비네팅"""
    result = image.copy()
    h, w = image.shape[:2]

    # 차가운 톤
    result = result.astype(np.float32)
    result[:, :, 0] *= 1.03  # Blue 약간 증가
    result[:, :, 2] *= 0.97  # Red 약간 감소
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 비네팅
    Y, X = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_dist = np.sqrt(cx ** 2 + cy ** 2)
    vignette = 1 - (dist / max_dist) * 0.25
    vignette = np.clip(vignette, 0, 1)[:, :, np.newaxis]

    result = (result.astype(np.float32) * vignette).astype(np.uint8)
    return result


def draw_mugshot_overlay(image: np.ndarray) -> np.ndarray:
    """머그샷 배경 오버레이"""
    result = image.copy()
    h, w = image.shape[:2]

    # 차가운 오버레이
    cold = np.full_like(image, (175, 160, 150), dtype=np.uint8)  # 차가운 회색
    result = cv2.addWeighted(result, 0.94, cold, 0.06, 0)

    # 키 측정 라인 (오른쪽)
    line_x = int(w * 0.95)
    line_count = 12
    for i in range(line_count):
        y = int(h * 0.08 + (h * 0.84 / line_count) * i)
        line_len = int(w * 0.04 if i % 2 == 0 else w * 0.025)
        cv2.line(result, (line_x - line_len, y), (line_x, y),
                  (200, 200, 200), 1, cv2.LINE_AA)
        if i % 2 == 0:
            cm_val = 190 - i * 8
            cv2.putText(result, str(cm_val),
                         (line_x - line_len - 30, y + 4),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 하단 번호판
    plate_h = int(h * 0.08)
    plate_y = int(h * 0.9)
    cv2.rectangle(result, (int(w * 0.15), plate_y),
                   (int(w * 0.85), plate_y + plate_h),
                   (0, 0, 0), -1)
    cv2.rectangle(result, (int(w * 0.15), plate_y),
                   (int(w * 0.85), plate_y + plate_h),
                   (200, 200, 200), 1)

    # 텍스트 (날짜)
    date_str = datetime.now().strftime("%Y.%m.%d")
    text = f"DRUG PREVENTION | {date_str}"
    font_scale = w / 1400
    cv2.putText(result, text,
                 (int(w * 0.3), plate_y + int(plate_h * 0.65)),
                 cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 정면 플래시 (밝은 중심 그라데이션)
    flash = np.zeros_like(image, dtype=np.float32)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - w // 2) ** 2 + (Y - int(h * 0.35)) ** 2)
    max_d = w * 0.5
    flash_mask = np.clip(1 - dist / max_d, 0, 1)[:, :, np.newaxis]
    result = np.clip(result.astype(np.float32) + flash_mask * 10, 0, 255).astype(np.uint8)

    return result


# ============================================
# 전체 필터 적용 파이프라인
# ============================================
def apply_all_filters(image: np.ndarray, landmarks, settings: dict) -> np.ndarray:
    """모든 AFTER 필터를 순차 적용"""
    result = image.copy()
    h, w = image.shape[:2]

    # 얼굴 마스크 생성
    face_mask = create_face_mask(landmarks, w, h)

    # 1. 피부톤 회색화
    if settings["skinGray"] > 0:
        result = apply_skin_gray(result, face_mask, settings["skinGray"] / 100)

    # 2. 피부 거칠기
    if settings["skinRough"] > 0:
        result = apply_skin_roughness(result, face_mask, settings["skinRough"] / 100)

    # 3. 다크서클
    if settings["darkCircle"] > 0:
        result = draw_dark_circles(result, landmarks, settings["darkCircle"] / 100)

    # 4. 눈 충혈
    if settings["eyeRedness"] > 0:
        result = draw_eye_redness(result, landmarks, settings["eyeRedness"] / 100)

    # 5. 멍한 눈
    if settings["dullEyes"] > 0:
        result = draw_dull_eyes(result, landmarks, settings["dullEyes"] / 100)

    # 6. 입 주변 착색
    if settings["mouthStain"] > 0:
        result = draw_mouth_stain(result, landmarks, settings["mouthStain"] / 100)

    # 7. 볼 꺼짐
    if settings["cheekHollow"] > 0:
        result = draw_cheek_hollow(result, landmarks, settings["cheekHollow"] / 100)

    # 8. 입꼬리 하강
    if settings["mouthDroop"] > 0:
        result = draw_mouth_droop(result, landmarks, settings["mouthDroop"] / 100)

    # 9. 비대칭
    if settings["asymmetry"] > 0:
        result = draw_asymmetry(result, landmarks, settings["asymmetry"] / 100)

    # 10. 치아 손상
    if settings["teethDamage"] > 0:
        result = draw_teeth_damage(result, landmarks, settings["teethDamage"] / 100)

    # 머그샷 배경
    if settings["mugshot"]:
        result = draw_mugshot_overlay(result)

    # 색보정
    result = apply_color_grading(result)

    return result


# ============================================
# 합성 이미지 생성 (다운로드/인쇄용)
# ============================================
def generate_composite(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """BEFORE + AFTER 합성 이미지 생성"""
    bh, bw = before.shape[:2]

    gap = 20
    header_h = 80
    footer_h = 60
    total_w = bw * 2 + gap * 3
    total_h = bh + header_h + footer_h + gap * 2

    # 어두운 배경
    composite = np.full((total_h, total_w, 3), (15, 10, 10), dtype=np.uint8)

    # BEFORE 이미지 배치
    y_start = header_h + gap
    composite[y_start:y_start + bh, gap:gap + bw] = before

    # AFTER 이미지 배치
    composite[y_start:y_start + bh, bw + gap * 2:bw + gap * 2 + bw] = after

    # BEFORE 테두리 (파란색)
    cv2.rectangle(composite, (gap - 2, y_start - 2),
                   (gap + bw + 1, y_start + bh + 1), (219, 152, 52), 2)
    # AFTER 테두리 (빨간색)
    cv2.rectangle(composite, (bw + gap * 2 - 2, y_start - 2),
                   (bw + gap * 2 + bw + 1, y_start + bh + 1), (70, 57, 230), 2)

    # BEFORE 라벨
    cv2.rectangle(composite, (gap + 8, y_start + 8),
                   (gap + 108, y_start + 38), (219, 152, 52), -1)
    cv2.putText(composite, "BEFORE", (gap + 16, y_start + 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # AFTER 라벨
    cv2.rectangle(composite, (bw + gap * 2 + 8, y_start + 8),
                   (bw + gap * 2 + 98, y_start + 38), (70, 57, 230), -1)
    cv2.putText(composite, "AFTER", (bw + gap * 2 + 18, y_start + 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 화살표
    arrow_x = total_w // 2
    arrow_y = y_start + bh // 2
    cv2.arrowedLine(composite, (arrow_x - 15, arrow_y), (arrow_x + 15, arrow_y),
                     (150, 150, 150), 2, tipLength=0.5)

    return composite


def image_to_download_bytes(image: np.ndarray) -> bytes:
    """numpy 이미지 → PNG 바이트"""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", quality=95)
    return buf.getvalue()


def draw_debug_overlay(image: np.ndarray, landmarks, quality: dict) -> np.ndarray:
    """디버그 오버레이 - 랜드마크 점 표시"""
    result = image.copy()
    h, w = image.shape[:2]

    # 모든 랜드마크 점
    for i, lm in enumerate(landmarks):
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(result, (x, y), 1, (0, 255, 0), -1)

    # 얼굴 중심
    nx, ny = get_landmark_point(landmarks, LM.NOSE_TIP, w, h)
    cv2.circle(result, (nx, ny), 6, (0, 255, 255), 2)

    # 정보 텍스트
    info_lines = [
        f"Face size: {quality.get('face_size', 0):.3f}",
        f"Yaw: {quality.get('yaw', 0):.1f}°",
        f"Brightness: {quality.get('brightness', 0):.0f}",
        f"Passed: {quality.get('all_passed', False)}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(result, line, (10, 25 + i * 22),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return result


# ============================================
# UI 화면
# ============================================

def render_start_screen():
    """시작 화면"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div class="start-container">
        <div class="start-badge">마약류 오남용 예방 캠페인</div>
        <div class="start-title">마약 <span class="accent">2컷</span></div>
        <div class="start-subtitle">약물이 바꿔버린 미래</div>
        <div class="start-desc">
            카메라로 얼굴을 촬영하면<br>
            <strong>약물 사용 전과 후</strong>의 모습을 비교할 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📸  체험 시작하기", type="primary", use_container_width=True):
            go_to_screen("camera")
            st.rerun()

    st.markdown("""
    <div class="privacy-notice" style="text-align:center; padding: 1rem;">
        🔒 촬영 이미지는 저장되지 않으며, 결과 확인 후 화면을 나가면 자동으로 삭제됩니다.
    </div>
    """, unsafe_allow_html=True)

    # 관리자 버튼 (하단 작게)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([3, 1, 3])
    with col2:
        if st.button("⚙️", key="admin_btn_start", help="관리자 설정"):
            go_to_screen("admin")
            st.rerun()


def render_camera_screen():
    """촬영 화면"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding: 0.3rem 0;">
        <span style="font-family:'Black Han Sans'; font-size:1.2rem; color:#f1f1f4;">
            마약 2컷 <span style="color:#e63946; font-size:0.7em;">: 약물이 바꿔버린 미래</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 안내 메시지
    st.markdown("""
    <div class="guide-msg warning">
        💡 얼굴이 화면 중앙에 오도록 촬영해주세요. 정면을 바라보면 더 좋은 결과를 얻을 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

    # 카메라 입력
    camera_photo = st.camera_input(
        "촬영 버튼을 눌러 얼굴을 촬영하세요",
        key=f"camera_{st.session_state.camera_key}",
    )

    if camera_photo is not None:
        # 이미지 로드
        file_bytes = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("이미지를 불러올 수 없습니다. 다시 촬영해주세요.")
            return

        # 좌우 반전 (셀카 모드)
        image = cv2.flip(image, 1)

        # 얼굴 감지 및 품질 검사
        with st.spinner("얼굴을 인식하는 중..."):
            landmarks, quality = detect_face(image)

        st.session_state.quality_info = quality

        if landmarks is None or not quality["all_passed"]:
            # 품질 미달 - 경고 표시
            msg_class = "error" if landmarks is None else "warning"
            st.markdown(f"""
            <div class="guide-msg {msg_class}">
                ⚠️ {quality["message"]}
            </div>
            """, unsafe_allow_html=True)

            # 디버그 모드
            if st.session_state.settings.get("debug"):
                if landmarks is not None:
                    debug_img = draw_debug_overlay(image, landmarks, quality)
                    st.image(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB),
                              caption="디버그: 랜드마크 오버레이", use_container_width=True)
                st.markdown(f"""
                <div class="debug-box">
                    감지: {quality['detected']} | 중앙: {quality['center_ok']} |
                    크기: {quality['size_ok']} ({quality['face_size']:.3f}) |
                    정면: {quality['frontal_ok']} (Yaw: {quality['yaw']:.1f}°) |
                    밝기: {quality['brightness_ok']} ({quality['brightness']:.0f})
                </div>
                """, unsafe_allow_html=True)

            if not quality["all_passed"]:
                st.warning("조건이 충족되지 않았습니다. 얼굴을 올바르게 위치시킨 후 다시 촬영해주세요.")

                # 조건 미달이어도 강제 진행 옵션
                if landmarks is not None:
                    if st.button("⚡ 현재 사진으로 진행하기", key="force_proceed"):
                        _process_image(image, landmarks)
                        st.rerun()
                return

        # 품질 통과 - 필터 적용 진행
        st.markdown("""
        <div class="guide-msg success">
            ✅ 얼굴 인식이 완료되었습니다! 필터를 적용합니다...
        </div>
        """, unsafe_allow_html=True)

        _process_image(image, landmarks)
        st.rerun()

    # 하단 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 처음으로", use_container_width=True):
            go_to_screen("start")
            st.rerun()
    with col2:
        if st.button("⚙️ 관리자 설정", use_container_width=True):
            go_to_screen("admin")
            st.rerun()


def _process_image(image, landmarks):
    """이미지 처리 및 필터 적용"""
    # BEFORE 저장 (메모리만)
    st.session_state.before_image = image.copy()
    st.session_state.landmarks = landmarks

    # AFTER 필터 적용
    after_image = apply_all_filters(image, landmarks, st.session_state.settings)
    st.session_state.after_image = after_image

    # 합성 이미지 생성
    st.session_state.composite_image = generate_composite(image, after_image)

    go_to_screen("result")


def render_result_screen():
    """결과 화면"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    before_img = st.session_state.before_image
    after_img = st.session_state.after_image
    composite_img = st.session_state.composite_image

    if before_img is None or after_img is None:
        st.warning("결과 이미지가 없습니다. 다시 촬영해주세요.")
        if st.button("📸 다시 촬영"):
            go_to_screen("camera")
            st.rerun()
        return

    # 헤더
    st.markdown('<div class="result-header">약물이 바꿔버린 미래</div>', unsafe_allow_html=True)

    # BEFORE / AFTER 비교
    col_b, col_a = st.columns(2)

    with col_b:
        st.markdown('<div class="label-before">BEFORE</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    with col_a:
        st.markdown('<div class="label-after">AFTER</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB), use_container_width=True)

    # 메시지
    st.markdown(
        '<div class="result-message">한 번의 선택이 건강과 일상을 무너뜨릴 수 있습니다.</div>',
        unsafe_allow_html=True
    )

    # 버튼 행
    btn_cols = st.columns(5)

    # 다운로드 버튼
    with btn_cols[0]:
        if composite_img is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="💾 저장",
                data=image_to_download_bytes(composite_img),
                file_name=f"마약2컷_{ts}.png",
                mime="image/png",
                use_container_width=True,
            )

    # AFTER만 다운로드
    with btn_cols[1]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📷 AFTER 저장",
            data=image_to_download_bytes(after_img),
            file_name=f"마약2컷_AFTER_{ts}.png",
            mime="image/png",
            use_container_width=True,
        )

    # 인쇄 (JavaScript)
    with btn_cols[2]:
        if st.button("🖨️ 인쇄", use_container_width=True):
            if composite_img is not None:
                b64 = base64.b64encode(image_to_download_bytes(composite_img)).decode()
                print_js = f"""
                <script>
                    var win = window.open('', '_blank');
                    win.document.write('<html><body style="margin:0;display:flex;justify-content:center;align-items:center;min-height:100vh;background:#fff;">');
                    win.document.write('<img src="data:image/png;base64,{b64}" style="max-width:100%;height:auto;" />');
                    win.document.write('</body></html>');
                    win.document.close();
                    setTimeout(function(){{ win.print(); }}, 500);
                </script>
                """
                st.components.v1.html(print_js, height=0)

    # 다시 촬영
    with btn_cols[3]:
        if st.button("🔄 다시 촬영", use_container_width=True):
            go_to_screen("camera")
            st.rerun()

    # 처음으로
    with btn_cols[4]:
        if st.button("🏠 처음으로", use_container_width=True):
            go_to_screen("start")
            st.rerun()

    # 개인정보 안내
    st.markdown("""
    <div class="privacy-notice" style="text-align:center;">
        🔒 촬영 이미지는 저장되지 않으며, 화면을 나가면 자동으로 삭제됩니다.
    </div>
    """, unsafe_allow_html=True)


def render_admin_screen():
    """관리자 설정 화면"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown('<div class="admin-header">⚙️ 관리자 설정</div>', unsafe_allow_html=True)

    # 인증
    if not st.session_state.admin_auth:
        pw = st.text_input("관리자 비밀번호", type="password", key="admin_pw_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("확인", type="primary", use_container_width=True):
                if pw == ADMIN_PASSWORD:
                    st.session_state.admin_auth = True
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        with col2:
            if st.button("돌아가기", use_container_width=True):
                prev = "camera" if st.session_state.before_image is not None else "start"
                go_to_screen(prev)
                st.rerun()
        return

    # 프리셋
    st.subheader("프리셋")
    preset_cols = st.columns(3)
    for i, (name, values) in enumerate(PRESETS.items()):
        with preset_cols[i]:
            if st.button(f"📋 {name}", use_container_width=True, key=f"preset_{name}"):
                for k, v in values.items():
                    st.session_state.settings[k] = v
                st.rerun()

    st.divider()

    # 필터 강도 슬라이더
    st.subheader("필터 강도")

    filter_labels = {
        "skinGray": "피부톤 회색화",
        "darkCircle": "다크서클",
        "eyeRedness": "눈 충혈",
        "dullEyes": "멍한 눈",
        "skinRough": "피부 거칠기",
        "mouthStain": "입 주변 착색",
        "cheekHollow": "볼 꺼짐",
        "mouthDroop": "입꼬리 하강",
        "asymmetry": "얼굴 비대칭",
        "teethDamage": "치아 손상",
    }

    col1, col2 = st.columns(2)
    items = list(filter_labels.items())
    for i, (key, label) in enumerate(items):
        with (col1 if i < len(items) // 2 else col2):
            st.session_state.settings[key] = st.slider(
                label, 0, 100,
                value=st.session_state.settings[key],
                key=f"slider_{key}"
            )

    st.divider()

    # 토글 설정
    st.subheader("기능 설정")

    st.session_state.settings["mugshot"] = st.checkbox(
        "머그샷 배경",
        value=st.session_state.settings["mugshot"],
        key="cb_mugshot"
    )
    st.session_state.settings["autoCapture"] = st.checkbox(
        "자동 촬영 기본값",
        value=st.session_state.settings["autoCapture"],
        key="cb_auto"
    )
    st.session_state.settings["debug"] = st.checkbox(
        "디버그 모드",
        value=st.session_state.settings["debug"],
        key="cb_debug"
    )

    st.divider()

    # 저장/초기화
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("💾 설정 저장", type="primary", use_container_width=True):
            st.success("설정이 저장되었습니다.")
    with col2:
        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.settings = {**DEFAULT_SETTINGS}
            st.rerun()
    with col3:
        if st.button("◀ 돌아가기", use_container_width=True):
            prev_screen = "camera" if st.session_state.before_image is not None else "start"
            st.session_state.screen = prev_screen
            st.rerun()

    # 현재 설정값 표시
    with st.expander("현재 설정값 (JSON)"):
        st.json(st.session_state.settings)


# ============================================
# 메인 라우팅
# ============================================
def main():
    """메인 앱 라우터"""
    init_session_state()

    screen = st.session_state.screen

    if screen == "start":
        render_start_screen()
    elif screen == "camera":
        render_camera_screen()
    elif screen == "result":
        render_result_screen()
    elif screen == "admin":
        render_admin_screen()
    else:
        render_start_screen()


if __name__ == "__main__":
    main()
