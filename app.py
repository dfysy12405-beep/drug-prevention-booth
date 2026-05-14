"""
마약 2컷: 약물이 바꿔버린 미래
================================
마약류 및 약물 오남용 예방 캠페인용 AI 필터 포토부스
Python + Streamlit + OpenCV + MediaPipe FaceMesh 기반

실행: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import time
from datetime import datetime

# ============================================================
# 1. 페이지 설정 (반드시 최상단)
# ============================================================
st.set_page_config(
    page_title="마약 2컷: 약물이 바꿔버린 미래",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# 2. 커스텀 CSS
# ============================================================
def inject_css():
    st.markdown("""
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&family=Black+Han+Sans&display=swap');

    /* ── 전역 ── */
    .stApp {
        background: #0a0a0f !important;
        font-family: 'Noto Sans KR', sans-serif !important;
    }
    [data-testid="stSidebar"] { display: none; }
    header[data-testid="stHeader"] { background: transparent !important; }
    [data-testid="stBottomBlockContainer"] { background: #0a0a0f !important; }

    /* ── 시작 화면 ── */
    .hero { text-align: center; padding: 1.5rem 1rem 1rem; max-width: 560px; margin: 0 auto; }
    .hero-badge {
        display: inline-block; padding: .3rem .9rem;
        background: rgba(230,57,70,.12); border: 1px solid rgba(230,57,70,.28);
        border-radius: 999px; font-size: .78rem; font-weight: 500;
        color: #e63946; letter-spacing: .04em; margin-bottom: 1.2rem;
    }
    .hero-title {
        font-family: 'Black Han Sans', sans-serif;
        font-size: clamp(2.8rem, 9vw, 5rem);
        line-height: 1.05; color: #f1f1f4; margin: 0 0 .2rem;
    }
    .hero-title b { color: #e63946; }
    .hero-sub {
        font-family: 'Black Han Sans', sans-serif;
        font-size: clamp(1rem, 3.5vw, 1.5rem);
        color: #8a8a9e; margin-bottom: 1.8rem;
    }
    .hero-desc { font-size: .92rem; color: #8a8a9e; line-height: 1.7; margin-bottom: 1.8rem; }
    .hero-desc strong { color: #f1f1f4; }
    .privacy {
        font-size: .72rem; color: #55556a; line-height: 1.5;
        margin-top: 1.2rem; text-align: center;
    }

    /* ── 안내 메시지 ── */
    .msg {
        text-align: center; padding: .65rem .8rem; border-radius: 10px;
        font-size: .88rem; font-weight: 500; margin: .4rem 0;
    }
    .msg-ok   { background: rgba(46,204,113,.08); border: 1px solid rgba(46,204,113,.25); color: #2ecc71; }
    .msg-warn { background: rgba(243,156,18,.08); border: 1px solid rgba(243,156,18,.25); color: #f39c12; }
    .msg-err  { background: rgba(231,76,60,.08);  border: 1px solid rgba(231,76,60,.25);  color: #e74c3c; }

    /* ── 결과 라벨 ── */
    .lbl {
        display: inline-block; padding: .18rem .7rem;
        border-radius: 7px; font-size: .78rem; font-weight: 700;
        letter-spacing: .07em; color: #fff; margin-bottom: .3rem;
    }
    .lbl-b { background: #3498db; }
    .lbl-a { background: #e63946; }
    .result-title {
        text-align: center; font-family: 'Black Han Sans', sans-serif;
        font-size: clamp(1.15rem, 3.5vw, 1.7rem); color: #f1f1f4;
        margin-bottom: .3rem;
    }
    .result-msg {
        text-align: center; font-size: .82rem; color: #8a8a9e;
        margin-bottom: .8rem; line-height: 1.5;
    }

    /* ── 디버그 ── */
    .dbg {
        background: rgba(0,0,0,.55); border: 1px solid #2a2a3e;
        border-radius: 8px; padding: .45rem .65rem;
        font-family: 'Courier New', monospace; font-size: .72rem;
        color: #2ecc71; line-height: 1.55;
    }

    /* ── 버튼 통일 ── */
    .stButton > button {
        font-family: 'Noto Sans KR', sans-serif !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: .5rem 1.2rem !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    .stButton > button:active { transform: scale(.97) !important; }

    /* 큰 시작 버튼 */
    .big-btn .stButton > button {
        font-size: 1.15rem !important;
        padding: .85rem 2rem !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px rgba(230,57,70,.25) !important;
    }

    /* ── 카메라 ── */
    [data-testid="stCameraInput"] { border-radius: 12px; overflow: hidden; }

    /* ── 슬라이더 색상 ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #e63946 !important;
    }

    /* ── 반응형 세로모드 ── */
    @media (max-width: 768px) {
        .hero-title { font-size: clamp(2.2rem, 12vw, 3.5rem); }
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# 3. 상수 / 설정
# ============================================================

# MediaPipe FaceMesh 랜드마크 인덱스
NOSE_TIP = 1
L_EYE_OUTER, R_EYE_OUTER = 33, 263
L_EYE_TOP, L_EYE_BOT = 159, 145
R_EYE_TOP, R_EYE_BOT = 386, 374
MOUTH_L, MOUTH_R = 61, 291
MOUTH_TOP, MOUTH_BOT = 13, 14
LIP_TOP, LIP_BOT = 0, 17
L_CHEEK, R_CHEEK = 234, 454
CHIN, FOREHEAD = 152, 10
L_TEMPLE, R_TEMPLE = 127, 356

FACE_OVAL = [
    10,338,297,332,284,251,389,356,454,323,361,288,
    397,365,379,378,400,377,152,148,176,149,150,136,
    172,58,132,93,234,127,162,21,54,103,67,109,
]
L_UNDER_EYE = [111,117,118,119,120,121,128]
R_UNDER_EYE = [340,346,347,348,349,350,357]
L_EYE_PTS = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246]
R_EYE_PTS = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
LIPS_OUTER = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185]

DEFAULT_SETTINGS = dict(
    skinGray=60, darkCircle=55, eyeRedness=45, dullEyes=40,
    skinRough=50, mouthStain=50, cheekHollow=45, mouthDroop=35,
    asymmetry=25, teethDamage=50, mugshot=True, debug=False,
)
PRESETS = {
    "약하게": dict(skinGray=35,darkCircle=30,eyeRedness=25,dullEyes=20,skinRough=25,
                   mouthStain=25,cheekHollow=20,mouthDroop=15,asymmetry=10,teethDamage=25),
    "보통":   dict(skinGray=60,darkCircle=55,eyeRedness=45,dullEyes=40,skinRough=50,
                   mouthStain=50,cheekHollow=45,mouthDroop=35,asymmetry=25,teethDamage=50),
    "강하게": dict(skinGray=85,darkCircle=80,eyeRedness=70,dullEyes=65,skinRough=75,
                   mouthStain=75,cheekHollow=70,mouthDroop=55,asymmetry=40,teethDamage=75),
}
ADMIN_PW = "1234"

FILTER_LABELS = {
    "skinGray":"피부톤 회색화", "darkCircle":"다크서클", "eyeRedness":"눈 충혈",
    "dullEyes":"멍한 눈", "skinRough":"피부 거칠기", "mouthStain":"입 주변 착색",
    "cheekHollow":"볼 꺼짐", "mouthDroop":"입꼬리 하강",
    "asymmetry":"얼굴 비대칭", "teethDamage":"치아 손상",
}


# ============================================================
# 4. 세션 상태
# ============================================================
def init_state():
    defs = dict(
        page="start", settings={**DEFAULT_SETTINGS},
        img_before=None, img_after=None, img_comp=None,
        landmarks=None, quality=None,
        admin_ok=False, cam_key=0,
    )
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v


def clear_images():
    """개인정보 보호: 이미지 즉시 삭제"""
    for k in ("img_before", "img_after", "img_comp", "landmarks", "quality"):
        st.session_state[k] = None


def goto(page):
    if page in ("start", "camera"):
        clear_images()
    st.session_state.page = page
    if page == "camera":
        st.session_state.cam_key += 1


# ============================================================
# 5. MediaPipe 싱글톤
# ============================================================
@st.cache_resource
def _face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1,
        refine_landmarks=True, min_detection_confidence=0.5,
    )


# ============================================================
# 6. 유틸리티
# ============================================================
def pt(lm_list, idx, w, h):
    """랜드마크 → 픽셀 좌표"""
    lm = lm_list[idx]
    return int(lm.x * w), int(lm.y * h)


def pts_center(lm_list, indices, w, h):
    """여러 랜드마크의 중심"""
    xs = [lm_list[i].x * w for i in indices]
    ys = [lm_list[i].y * h for i in indices]
    return int(np.mean(xs)), int(np.mean(ys))


def eye_dist(lm, w, h):
    lx, ly = pt(lm, L_EYE_OUTER, w, h)
    rx, ry = pt(lm, R_EYE_OUTER, w, h)
    return np.hypot(lx - rx, ly - ry)


def to_png(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG", quality=95)
    return buf.getvalue()


# ============================================================
# 7. 얼굴 감지 + 품질 검사
# ============================================================
def detect_face(image):
    """
    Returns (landmarks, quality_dict)
    landmarks 는 mediapipe landmark 리스트, 실패 시 None
    """
    fm = _face_mesh()
    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)

    q = dict(detected=False, center=False, size=False,
             frontal=False, bright=False, ok=False,
             msg="", face_pct=0, yaw=0, bri=0)

    if not res.multi_face_landmarks:
        q["msg"] = "😶 얼굴이 감지되지 않았습니다. 카메라를 바라봐주세요."
        return None, q

    lm = res.multi_face_landmarks[0].landmark
    q["detected"] = True

    nose = lm[NOSE_TIP]
    cx, cy = nose.x, nose.y

    # 중앙
    q["center"] = 0.2 < cx < 0.8 and 0.12 < cy < 0.78

    # 크기
    le = lm[L_EYE_OUTER]; re = lm[R_EYE_OUTER]
    ed = np.hypot(le.x - re.x, le.y - re.y)
    q["face_pct"] = ed
    small = ed < 0.11; big = ed > 0.48
    q["size"] = not small and not big

    # 정면
    nb = lm[6]; lt = lm[L_TEMPLE]; rt = lm[R_TEMPLE]
    fw = abs(rt.x - lt.x) or 1e-6
    yaw = ((nb.x - lt.x) / fw - 0.5) * 90
    q["yaw"] = yaw
    q["frontal"] = abs(yaw) < 18

    # 밝기
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bri = float(np.mean(gray))
    q["bri"] = bri
    q["bright"] = bri > 45

    q["ok"] = all([q["center"], q["size"], q["frontal"], q["bright"]])

    if not q["center"]:
        q["msg"] = "↔️ 얼굴을 화면 중앙에 맞춰주세요."
    elif small:
        q["msg"] = "🔍 얼굴이 너무 작습니다. 조금 가까이 와주세요."
    elif big:
        q["msg"] = "🔎 얼굴이 너무 큽니다. 조금 뒤로 이동해주세요."
    elif not q["frontal"]:
        q["msg"] = "👤 정면을 바라봐주세요."
    elif not q["bright"]:
        q["msg"] = "💡 조명이 어두워 얼굴 인식이 어렵습니다."
    else:
        q["msg"] = "✅ 좋습니다! 얼굴 인식이 완료되었습니다."

    return lm, q


# ============================================================
# 8. 얼굴 마스크
# ============================================================
def face_mask(lm, w, h):
    mask = np.zeros((h, w), np.uint8)
    pts = np.array([[int(lm[i].x * w), int(lm[i].y * h)] for i in FACE_OVAL], np.int32)
    cv2.fillPoly(mask, [pts], 255)
    mask = cv2.GaussianBlur(mask, (21, 21), 10)
    return mask


# ============================================================
# 9. 필터 함수들
# ============================================================

def f_skin_gray(img, mask, t):
    """피부톤 회색화: 채도 감소 + 누런빛"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= (1 - t * 0.72)       # 채도 감소
    hsv[:,:,0] = np.clip(hsv[:,:,0] + t * 5, 0, 180)  # 약간 노랗게
    hsv[:,:,2] *= (1 - t * 0.08)       # 명도 살짝 감소
    hsv = np.clip(hsv, 0, [180, 255, 255]).astype(np.uint8)
    desat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    m3 = (mask / 255.0)[:,:,None]
    return np.clip(img * (1 - m3 * t) + desat * m3 * t, 0, 255).astype(np.uint8)


def f_skin_rough(img, mask, t):
    """피부 거칠기: 노이즈 + 대비"""
    noise = (np.random.randn(*img.shape) * t * 18).astype(np.float32)
    m3 = (mask / 255.0)[:,:,None]
    out = np.clip(img.astype(np.float32) + noise * m3, 0, 255).astype(np.uint8)
    # CLAHE 대비
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    cl = cv2.createCLAHE(clipLimit=1.0 + t * 1.5, tileGridSize=(8, 8))
    eg = cl.apply(gray)
    diff = eg.astype(np.float32) - gray.astype(np.float32)
    for c in range(3):
        ch = out[:,:,c].astype(np.float32) + diff * m3[:,:,0] * t * 0.3
        out[:,:,c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def _ov(img, overlay, alpha):
    """오버레이 합성 헬퍼"""
    return cv2.addWeighted(img, 1.0, overlay, alpha, 0)


def f_dark_circles(img, lm, t):
    """다크서클"""
    h, w = img.shape[:2]; ed = eye_dist(lm, w, h)
    ov = np.zeros_like(img)
    for ep in (L_UNDER_EYE, R_UNDER_EYE):
        cx, cy = pts_center(lm, ep, w, h)
        rx, ry = int(ed * 0.19), int(ed * 0.09)
        cv2.ellipse(ov, (cx, cy + ry // 2), (rx, ry), 0, 0, 360, (30, 15, 45), -1)
    ks = max(3, int(ed * 0.08)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), ed * 0.04)
    return _ov(img, ov, t * 0.5)


def f_eye_red(img, lm, t):
    """눈 충혈"""
    h, w = img.shape[:2]; ed = eye_dist(lm, w, h)
    ov = np.zeros_like(img)
    for ep in (L_EYE_PTS, R_EYE_PTS):
        cx, cy = pts_center(lm, ep, w, h)
        r = int(ed * 0.1)
        cv2.ellipse(ov, (cx, cy), (int(r*1.2), int(r*.7)), 0, 0, 360, (20, 30, 175), -1)
        cv2.ellipse(ov, (cx, cy), (int(r*.8), int(r*.45)), 0, 0, 360, (10, 90, 150), -1)
    ks = max(3, int(ed * 0.06)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), ed * 0.03)
    return _ov(img, ov, t * 0.3)


def f_dull_eyes(img, lm, t):
    """멍한 눈: 눈 주위 그림자"""
    h, w = img.shape[:2]; ed = eye_dist(lm, w, h)
    ov = np.zeros_like(img)
    for ep in (L_EYE_PTS, R_EYE_PTS):
        cx, cy = pts_center(lm, ep, w, h)
        r = int(ed * 0.2)
        cv2.ellipse(ov, (cx, cy - int(r*.3)), (int(r*1.3), r), 0, 0, 360, (28, 12, 30), -1)
    ks = max(3, int(ed * 0.12)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), ed * 0.06)
    return _ov(img, ov, t * 0.3)


def f_mouth_stain(img, lm, t):
    """입 주변 착색"""
    h, w = img.shape[:2]; ov = np.zeros_like(img)
    mlx, mly = pt(lm, MOUTH_L, w, h)
    mrx, mry = pt(lm, MOUTH_R, w, h)
    mtx, mty = pt(lm, MOUTH_TOP, w, h)
    mbx, mby = pt(lm, MOUTH_BOT, w, h)
    cx, cy = (mlx+mrx)//2, (mty+mby)//2
    mw, mh = abs(mrx-mlx), abs(mby-mty)
    cv2.ellipse(ov, (cx, cy), (int(mw*.7), int(mh*1.5)), 0, 0, 360, (15, 15, 58), -1)
    cv2.ellipse(ov, (cx, cy), (int(mw*.45), int(mh*.8)), 0, 0, 360, (12, 10, 40), -1)
    ks = max(3, int(mw * 0.15)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), mw * 0.08)
    return _ov(img, ov, t * 0.42)


def f_cheek_hollow(img, lm, t):
    """볼 꺼짐"""
    h, w = img.shape[:2]; ed = eye_dist(lm, w, h)
    ov = np.zeros_like(img); r = int(ed * 0.18)
    lcx, lcy = pt(lm, L_CHEEK, w, h); mlx, mly = pt(lm, MOUTH_L, w, h)
    rcx, rcy = pt(lm, R_CHEEK, w, h); mrx, mry = pt(lm, MOUTH_R, w, h)
    c1 = (int(lcx*.6+mlx*.4), int(lcy*.4+mly*.6))
    c2 = (int(rcx*.6+mrx*.4), int(rcy*.4+mry*.6))
    cv2.ellipse(ov, c1, (int(r*1.2), int(r*.7)), -15, 0, 360, (22, 12, 25), -1)
    cv2.ellipse(ov, c2, (int(r*1.2), int(r*.7)),  15, 0, 360, (22, 12, 25), -1)
    ks = max(3, int(r * 1.4)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), r * 0.7)
    return _ov(img, ov, t * 0.38)


def f_mouth_droop(img, lm, t):
    """입꼬리 하강 그림자"""
    h, w = img.shape[:2]; ov = np.zeros_like(img)
    mlx, mly = pt(lm, MOUTH_L, w, h)
    mrx, mry = pt(lm, MOUTH_R, w, h)
    r = max(3, int(abs(mrx - mlx) * 0.06))
    cv2.ellipse(ov, (mlx, mly + r*2), (r*2, r*3), 0, 0, 360, (25, 12, 22), -1)
    cv2.ellipse(ov, (mrx, mry + r*2), (r*2, r*3), 0, 0, 360, (25, 12, 22), -1)
    ks = max(3, r * 3) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), r * 1.2)
    return _ov(img, ov, t * 0.42)


def f_asymmetry(img, lm, t):
    """약한 비대칭: 한쪽 음영"""
    h, w = img.shape[:2]; ov = np.zeros_like(img)
    nx, _ = pt(lm, NOSE_TIP, w, h)
    _, fy = pt(lm, FOREHEAD, w, h)
    _, cy = pt(lm, CHIN, w, h)
    cx = nx + int(w * 0.06)
    my = (fy + cy) // 2
    rx, ry = int(w * 0.14), int(abs(cy - fy) * 0.48)
    cv2.ellipse(ov, (cx, my), (rx, ry), 0, 0, 360, (18, 8, 18), -1)
    ks = max(3, int(rx * 0.8)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), rx * 0.4)
    return _ov(img, ov, t * 0.16)


def f_teeth_damage(img, lm, t):
    """치아 손상: 누런 변색 + 검은 틈"""
    h, w = img.shape[:2]; out = img.copy()
    mlx, _ = pt(lm, MOUTH_L, w, h); mrx, _ = pt(lm, MOUTH_R, w, h)
    _, mty  = pt(lm, MOUTH_TOP, w, h); _, mby = pt(lm, MOUTH_BOT, w, h)
    cx = (mlx + mrx) // 2; cy = (mty + mby) // 2
    mw = abs(mrx - mlx); mh = abs(mby - mty)

    # 어두운 입 안쪽
    ov = np.zeros_like(img)
    cv2.ellipse(ov, (cx, cy - int(mh*.12)), (int(mw*.22), int(mh*.28)), 0, 0, 360, (5,5,18), -1)
    ks = max(3, int(mw * 0.04)) | 1
    ov = cv2.GaussianBlur(ov, (ks, ks), mw * 0.02)
    out = _ov(out, ov, t * 0.55)

    # 누런 치아
    ty = cy - int(mh * 0.15)
    tw, th = int(mw * 0.06), int(mh * 0.22)
    rng = np.random.RandomState(42)
    for i in range(-2, 3):
        tx = cx + i * int(tw * 1.4)
        color = (int(28 + rng.random() * 18),
                 int(95 + rng.random() * 30),
                 int(135 + rng.random() * 40))
        cv2.rectangle(out, (tx - tw//2, ty - th//2), (tx + tw//2, ty + th//2), color, -1)
        if rng.random() < t * 0.5:
            gw = max(1, int(mw * 0.007))
            cv2.rectangle(out, (tx + tw//2 - gw, ty - th//2), (tx + tw//2, ty + th//2), (5,4,8), -1)

    # 블러
    y1, y2 = max(0, ty - th), min(h, ty + th)
    x1, x2 = max(0, cx - int(mw*.28)), min(w, cx + int(mw*.28))
    if y2 > y1 and x2 > x1:
        bk = max(3, int(mw * 0.025)) | 1
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (bk, bk), 0)
    return out


def f_mugshot(img):
    """머그샷 배경: 키측정선 + 번호판 + 차가운톤"""
    h, w = img.shape[:2]; out = img.copy()

    # 차가운 오버레이
    cold = np.full_like(img, (175, 160, 150), np.uint8)
    out = cv2.addWeighted(out, 0.94, cold, 0.06, 0)

    # 키 측정선 (우측)
    lx = int(w * 0.95)
    for i in range(12):
        y = int(h * 0.08 + (h * 0.84 / 12) * i)
        ll = int(w * (0.04 if i % 2 == 0 else 0.025))
        cv2.line(out, (lx - ll, y), (lx, y), (200, 200, 200), 1, cv2.LINE_AA)
        if i % 2 == 0:
            cv2.putText(out, str(190 - i * 8), (lx - ll - 28, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1)

    # 번호판
    ph = int(h * 0.08); py = int(h * 0.9)
    cv2.rectangle(out, (int(w*.15), py), (int(w*.85), py + ph), (0, 0, 0), -1)
    cv2.rectangle(out, (int(w*.15), py), (int(w*.85), py + ph), (200, 200, 200), 1)
    ds = datetime.now().strftime("%Y.%m.%d")
    fs = w / 1500
    cv2.putText(out, f"DRUG PREVENTION  |  {ds}",
                (int(w * 0.28), py + int(ph * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)

    # 정면 플래시
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((X - w//2)**2 + (Y - int(h*.35))**2).astype(np.float32)
    fl = np.clip(1 - d / (w * 0.5), 0, 1)[:,:,None]
    out = np.clip(out.astype(np.float32) + fl * 10, 0, 255).astype(np.uint8)
    return out


def f_color_grade(img):
    """색보정: 차가운 톤 + 비네팅"""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    out[:,:,0] *= 1.03  # Blue↑
    out[:,:,2] *= 0.97  # Red↓
    out = np.clip(out, 0, 255).astype(np.uint8)
    # 비네팅
    Y, X = np.ogrid[:h, :w]
    d = np.sqrt((X - w//2)**2 + (Y - h//2)**2)
    mx = np.sqrt((w//2)**2 + (h//2)**2)
    vig = np.clip(1 - (d / mx) * 0.25, 0, 1)[:,:,None]
    return (out.astype(np.float32) * vig).astype(np.uint8)


# ============================================================
# 10. 전체 필터 파이프라인
# ============================================================
def apply_filters(img, lm, s):
    """모든 필터를 순차 적용하여 AFTER 이미지 생성"""
    out = img.copy()
    h, w = img.shape[:2]
    mask = face_mask(lm, w, h)
    t = lambda k: s[k] / 100.0

    if s["skinGray"] > 0:     out = f_skin_gray(out, mask, t("skinGray"))
    if s["skinRough"] > 0:    out = f_skin_rough(out, mask, t("skinRough"))
    if s["darkCircle"] > 0:   out = f_dark_circles(out, lm, t("darkCircle"))
    if s["eyeRedness"] > 0:   out = f_eye_red(out, lm, t("eyeRedness"))
    if s["dullEyes"] > 0:     out = f_dull_eyes(out, lm, t("dullEyes"))
    if s["mouthStain"] > 0:   out = f_mouth_stain(out, lm, t("mouthStain"))
    if s["cheekHollow"] > 0:  out = f_cheek_hollow(out, lm, t("cheekHollow"))
    if s["mouthDroop"] > 0:   out = f_mouth_droop(out, lm, t("mouthDroop"))
    if s["asymmetry"] > 0:    out = f_asymmetry(out, lm, t("asymmetry"))
    if s["teethDamage"] > 0:  out = f_teeth_damage(out, lm, t("teethDamage"))
    if s["mugshot"]:          out = f_mugshot(out)
    out = f_color_grade(out)
    return out


# ============================================================
# 11. 합성 이미지 (다운로드용)
# ============================================================
def composite(before, after):
    bh, bw = before.shape[:2]
    gap, hdr, ftr = 16, 70, 52
    tw = bw * 2 + gap * 3
    th = bh + hdr + ftr + gap * 2
    c = np.full((th, tw, 3), (15, 10, 10), np.uint8)
    y0 = hdr + gap
    # BEFORE
    c[y0:y0+bh, gap:gap+bw] = before
    cv2.rectangle(c, (gap-2, y0-2), (gap+bw+1, y0+bh+1), (219,152,52), 2)
    cv2.rectangle(c, (gap+6, y0+6), (gap+90, y0+32), (219,152,52), -1)
    cv2.putText(c, "BEFORE", (gap+12, y0+26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    # AFTER
    ax = bw + gap * 2
    c[y0:y0+bh, ax:ax+bw] = after
    cv2.rectangle(c, (ax-2, y0-2), (ax+bw+1, y0+bh+1), (70,57,230), 2)
    cv2.rectangle(c, (ax+6, y0+6), (ax+82, y0+32), (70,57,230), -1)
    cv2.putText(c, "AFTER", (ax+14, y0+26), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
    # 화살표
    cv2.arrowedLine(c, (tw//2-12, y0+bh//2), (tw//2+12, y0+bh//2), (160,160,160), 2, tipLength=.5)
    return c


# ============================================================
# 12. 디버그 오버레이
# ============================================================
def debug_overlay(img, lm, q):
    out = img.copy()
    h, w = img.shape[:2]
    for l in lm:
        cv2.circle(out, (int(l.x*w), int(l.y*h)), 1, (0,255,0), -1)
    nx, ny = pt(lm, NOSE_TIP, w, h)
    col = (0,255,0) if q["ok"] else (0,0,255)
    cv2.circle(out, (nx, ny), 6, col, 2)
    lines = [
        f"Size: {q['face_pct']:.3f}",
        f"Yaw: {q['yaw']:.1f}",
        f"Bri: {q['bri']:.0f}",
        f"OK: {q['ok']}",
    ]
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (8, 22+i*20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,255,0), 1)
    return out


# ============================================================
# 13. 화면: 시작
# ============================================================
def page_start():
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">마약류 오남용 예방 캠페인</div>
        <div class="hero-title">마약 <b>2컷</b></div>
        <div class="hero-sub">약물이 바꿔버린 미래</div>
        <div class="hero-desc">
            카메라로 얼굴을 촬영하면<br>
            <strong>약물 사용 전과 후</strong>의 모습을 비교할 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, c, _ = st.columns([1, 2, 1])
    with c:
        st.markdown('<div class="big-btn">', unsafe_allow_html=True)
        if st.button("📸  체험 시작하기", type="primary", use_container_width=True):
            goto("camera"); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="privacy">🔒 촬영 이미지는 저장되지 않으며, 결과 확인 후 화면을 나가면 자동으로 삭제됩니다.</div>', unsafe_allow_html=True)

    # 관리자 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    _, c2, _ = st.columns([5, 1, 5])
    with c2:
        if st.button("⚙️", help="관리자 설정"):
            goto("admin"); st.rerun()


# ============================================================
# 14. 화면: 촬영
# ============================================================
def page_camera():
    st.markdown("""
    <div style="text-align:center; padding:.2rem 0 .4rem;">
        <span style="font-family:'Black Han Sans'; font-size:1.15rem; color:#f1f1f4;">
            마약 2컷 <span style="color:#e63946; font-size:.65em;">: 약물이 바꿔버린 미래</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="msg msg-warn">
        💡 얼굴이 화면 중앙에 오도록 촬영해주세요. 정면을 바라보면 더 좋은 결과를 얻을 수 있습니다.
    </div>
    """, unsafe_allow_html=True)

    photo = st.camera_input("촬영 버튼을 눌러주세요", key=f"cam_{st.session_state.cam_key}")

    if photo is not None:
        raw = np.asarray(bytearray(photo.read()), np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            st.error("이미지를 불러올 수 없습니다. 다시 촬영해주세요.")
            return

        img = cv2.flip(img, 1)  # 셀카 미러 해제

        with st.spinner("🔍 얼굴을 인식하는 중..."):
            lm, q = detect_face(img)

        # 디버그 모드
        if st.session_state.settings.get("debug") and lm is not None:
            dbg_img = debug_overlay(img, lm, q)
            st.image(cv2.cvtColor(dbg_img, cv2.COLOR_BGR2RGB),
                     caption="디버그 오버레이", use_container_width=True)
            st.markdown(f"""<div class="dbg">
                감지: {q['detected']} | 중앙: {q['center']} | 크기: {q['size']} ({q['face_pct']:.3f})<br>
                정면: {q['frontal']} (Yaw {q['yaw']:.1f}°) | 밝기: {q['bright']} ({q['bri']:.0f})
            </div>""", unsafe_allow_html=True)

        if lm is None:
            st.markdown(f'<div class="msg msg-err">{q["msg"]}</div>', unsafe_allow_html=True)
            st.info("다시 촬영해주세요.")
            return

        if not q["ok"]:
            st.markdown(f'<div class="msg msg-warn">{q["msg"]}</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.info("조건을 맞춰서 다시 촬영해주세요.")
            with c2:
                if st.button("⚡ 현재 사진으로 진행", use_container_width=True):
                    _process(img, lm); st.rerun()
            return

        st.markdown('<div class="msg msg-ok">✅ 얼굴 인식 완료! 필터를 적용합니다...</div>', unsafe_allow_html=True)
        _process(img, lm)
        st.rerun()

    # 하단 버튼
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🏠 처음으로", use_container_width=True):
            goto("start"); st.rerun()
    with c2:
        if st.button("⚙️ 관리자 설정", use_container_width=True):
            goto("admin"); st.rerun()


def _process(img, lm):
    """이미지 처리 → AFTER 생성 → 결과 페이지 이동"""
    st.session_state.img_before = img.copy()
    st.session_state.landmarks = lm
    after = apply_filters(img, lm, st.session_state.settings)
    st.session_state.img_after = after
    st.session_state.img_comp = composite(img, after)
    goto("result")


# ============================================================
# 15. 화면: 결과
# ============================================================
def page_result():
    bef = st.session_state.img_before
    aft = st.session_state.img_after
    comp = st.session_state.img_comp

    if bef is None or aft is None:
        st.warning("결과 이미지가 없습니다.")
        if st.button("📸 촬영하러 가기"):
            goto("camera"); st.rerun()
        return

    st.markdown('<div class="result-title">약물이 바꿔버린 미래</div>', unsafe_allow_html=True)

    # BEFORE / AFTER
    cb, ca = st.columns(2)
    with cb:
        st.markdown('<div class="lbl lbl-b">BEFORE</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(bef, cv2.COLOR_BGR2RGB), use_container_width=True)
    with ca:
        st.markdown('<div class="lbl lbl-a">AFTER</div>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(aft, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.markdown('<div class="result-msg">한 번의 선택이 건강과 일상을 무너뜨릴 수 있습니다.</div>', unsafe_allow_html=True)

    # 버튼
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    b1, b2, b3, b4 = st.columns(4)

    with b1:
        st.download_button("💾 비교 이미지 저장", data=to_png(comp),
                           file_name=f"마약2컷_{ts}.png", mime="image/png",
                           use_container_width=True)
    with b2:
        st.download_button("📷 AFTER 저장", data=to_png(aft),
                           file_name=f"마약2컷_AFTER_{ts}.png", mime="image/png",
                           use_container_width=True)
    with b3:
        if st.button("🔄 다시 촬영", use_container_width=True):
            goto("camera"); st.rerun()
    with b4:
        if st.button("🏠 처음으로", use_container_width=True):
            goto("start"); st.rerun()

    st.markdown('<div class="privacy">🔒 촬영 이미지는 저장되지 않으며, 화면을 나가면 자동으로 삭제됩니다.</div>', unsafe_allow_html=True)


# ============================================================
# 16. 화면: 관리자
# ============================================================
def page_admin():
    st.markdown("### ⚙️ 관리자 설정")

    if not st.session_state.admin_ok:
        pw = st.text_input("관리자 비밀번호", type="password", key="adm_pw")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("확인", type="primary", use_container_width=True):
                if pw == ADMIN_PW:
                    st.session_state.admin_ok = True; st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        with c2:
            if st.button("돌아가기", use_container_width=True):
                goto("start"); st.rerun()
        return

    # ─── 프리셋 ───
    st.markdown("**프리셋**")
    pc = st.columns(3)
    for i, (name, vals) in enumerate(PRESETS.items()):
        with pc[i]:
            if st.button(f"📋 {name}", use_container_width=True, key=f"pr_{name}"):
                for k, v in vals.items():
                    st.session_state.settings[k] = v
                st.rerun()

    st.divider()

    # ─── 슬라이더 ───
    st.markdown("**필터 강도**")
    items = list(FILTER_LABELS.items())
    c1, c2 = st.columns(2)
    for i, (key, label) in enumerate(items):
        with (c1 if i < len(items) // 2 else c2):
            st.session_state.settings[key] = st.slider(
                label, 0, 100, st.session_state.settings[key], key=f"sl_{key}")

    st.divider()

    # ─── 토글 ───
    st.markdown("**기능 설정**")
    st.session_state.settings["mugshot"] = st.checkbox(
        "머그샷 배경", st.session_state.settings["mugshot"], key="cb_mug")
    st.session_state.settings["debug"] = st.checkbox(
        "디버그 모드", st.session_state.settings["debug"], key="cb_dbg")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("💾 설정 저장", type="primary", use_container_width=True):
            st.success("✅ 저장 완료!")
    with c2:
        if st.button("🔄 초기화", use_container_width=True):
            st.session_state.settings = {**DEFAULT_SETTINGS}; st.rerun()
    with c3:
        if st.button("◀ 돌아가기", use_container_width=True):
            goto("start"); st.rerun()

    with st.expander("📄 현재 설정값 (JSON)"):
        st.json(st.session_state.settings)


# ============================================================
# 17. 메인
# ============================================================
def main():
    init_state()
    inject_css()

    pages = {
        "start": page_start,
        "camera": page_camera,
        "result": page_result,
        "admin": page_admin,
    }
    pages.get(st.session_state.page, page_start)()


if __name__ == "__main__":
    main()
