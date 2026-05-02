import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import gdown
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartPlate – Nutrition Analyzer",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_ID             = '1BesiFj-R5qQ--z0-_2kD8ZvNcbdGkppY'  # ⚠️ CHANGE THIS
MODEL_PATH           = 'best.pt'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD        = 0.45

CLASSES      = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
FOOD_CLASSES = ['buah', 'karbohidrat', 'protein', 'sayur']   # minuman EXCLUDED dari kalkulasi

# ==============================================================================
# NUTRITION DATABASE
# Referensi: FatSecret Indonesia (Nutrisi), Antarlina et al. (2009) & FAO/INFOODS (Densitas)
# ==============================================================================

NUTRITION_DB = {
    'buah': {
        'name': 'Buah', 'emoji': '🍎', 'color': '#E8544A',
        'density': 0.97,
        'kalori_per_100g': 51.7, 'protein_per_100g': 0.66,
        'karbohidrat_per_100g': 13.04, 'lemak_per_100g': 0.29, 'serat_per_100g': 1.60
    },
    'karbohidrat': {
        'name': 'Karbohidrat', 'emoji': '🍚', 'color': '#F5A623',
        'density': 0.73,
        'kalori_per_100g': 173.2, 'protein_per_100g': 5.32,
        'karbohidrat_per_100g': 33.06, 'lemak_per_100g': 2.24, 'serat_per_100g': 2.73
    },
    'minuman': {
        'name': 'Minuman', 'emoji': '🥤', 'color': '#4A90D9',
        'density': 1.04,
        'kalori_per_100g': 47.7, 'protein_per_100g': 1.39,
        'karbohidrat_per_100g': 9.24, 'lemak_per_100g': 0.79, 'serat_per_100g': 0
    },
    'protein': {
        'name': 'Protein', 'emoji': '🍗', 'color': '#C0392B',
        'density': 0.95,
        'kalori_per_100g': 200.4, 'protein_per_100g': 19.71,
        'karbohidrat_per_100g': 2.81, 'lemak_per_100g': 12.20, 'serat_per_100g': 0
    },
    'sayur': {
        'name': 'Sayur', 'emoji': '🥗', 'color': '#27AE60',
        'density': 0.50,
        'kalori_per_100g': 54.25, 'protein_per_100g': 2.31,
        'karbohidrat_per_100g': 10.03, 'lemak_per_100g': 0.86, 'serat_per_100g': 2.95
    }
}

# ==============================================================================
# IDEAL COMPOSITION – "ISI PIRINGKU"
# Referensi: Kemenkes RI. (2017). Pedoman Gizi Seimbang: Isi Piringku
# Minuman TIDAK termasuk pedoman proporsi piring
# ==============================================================================

IDEAL_COMPOSITION = {
    'karbohidrat': 35,
    'sayur':       35,
    'protein':     15,
    'buah':        15,
}

# ==============================================================================
# AKG DATABASE – Angka Kecukupan Gizi Indonesia 2019
# Referensi: Permenkes RI No. 28 Tahun 2019, Lampiran I Tabel 1
# ==============================================================================

AKG_DATABASE = {
    'male_19_29':          {'label': '👨 Laki-laki 19–29 tahun',   'kalori': 2650, 'protein': 65, 'karbohidrat': 430, 'lemak': 75, 'serat': 37},
    'male_30_49':          {'label': '👨 Laki-laki 30–49 tahun',   'kalori': 2550, 'protein': 65, 'karbohidrat': 415, 'lemak': 70, 'serat': 36},
    'male_50_64':          {'label': '👨 Laki-laki 50–64 tahun',   'kalori': 2150, 'protein': 65, 'karbohidrat': 340, 'lemak': 60, 'serat': 30},
    'female_19_29':        {'label': '👩 Perempuan 19–29 tahun',   'kalori': 2250, 'protein': 60, 'karbohidrat': 360, 'lemak': 65, 'serat': 32},
    'female_30_49':        {'label': '👩 Perempuan 30–49 tahun',   'kalori': 2150, 'protein': 60, 'karbohidrat': 340, 'lemak': 60, 'serat': 30},
    'female_50_64':        {'label': '👩 Perempuan 50–64 tahun',   'kalori': 1800, 'protein': 60, 'karbohidrat': 280, 'lemak': 50, 'serat': 25},
    'child_7_9':           {'label': '🧒 Anak 7–9 tahun',          'kalori': 1650, 'protein': 40, 'karbohidrat': 250, 'lemak': 55, 'serat': 23},
    'child_10_12':         {'label': '🧒 Anak 10–12 tahun',        'kalori': 1950, 'protein': 53, 'karbohidrat': 290, 'lemak': 65, 'serat': 28},
    'pregnant_trimester1': {'label': '🤰 Ibu Hamil Trimester 1',   'kalori': 2430, 'protein': 61, 'karbohidrat': 385, 'lemak': 67, 'serat': 35},
    'pregnant_trimester2': {'label': '🤰 Ibu Hamil Trimester 2',   'kalori': 2550, 'protein': 70, 'karbohidrat': 400, 'lemak': 67, 'serat': 36},
    'pregnant_trimester3': {'label': '🤰 Ibu Hamil Trimester 3',   'kalori': 2550, 'protein': 90, 'karbohidrat': 400, 'lemak': 67, 'serat': 36},
}

# ==============================================================================
# CUSTOM CSS  –  Clean & Classy
# ==============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.main > div { padding-top: 1.5rem; }

/* ── Brand Header ── */
.sp-header {
    display: flex; align-items: center; gap: 14px;
    padding: 1.6rem 2rem 1.2rem;
    background: linear-gradient(135deg, #1a3a1a 0%, #2d5a1b 60%, #3d7a25 100%);
    border-radius: 16px; margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(45,90,27,0.25);
}
.sp-header-icon { font-size: 2.8rem; }
.sp-header-text h1 {
    font-family: 'DM Serif Display', serif;
    color: #f0f7e8; margin: 0; font-size: 2rem; letter-spacing: -0.5px;
}
.sp-header-text p {
    color: #a8d87e; margin: 0; font-size: 0.88rem; font-weight: 500; letter-spacing: 0.5px;
}

/* ── Guide Cards ── */
.guide-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px; margin-bottom: 0.5rem;
}
.guide-card {
    background: #f8fdf4; border: 1.5px solid #d4e8c2;
    border-radius: 12px; padding: 14px 16px;
    display: flex; align-items: flex-start; gap: 10px;
}
.guide-card-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 1px; }
.guide-card-text { font-size: 0.83rem; color: #3a5c2a; line-height: 1.45; }
.guide-card-text strong { display: block; color: #1a3a0a; margin-bottom: 2px; font-size: 0.86rem; }

/* ── Disclaimer card ── */
.disclaimer-card {
    background: #fffbf0; border: 1.5px solid #f5d87a;
    border-left: 4px solid #d4a017; border-radius: 10px;
    padding: 12px 16px; margin-top: 8px; font-size: 0.82rem; color: #5a4a10;
    line-height: 1.5;
}
.disclaimer-card strong { color: #3a2e00; }

/* ── Balance Score Banner ── */
.score-banner {
    border-radius: 14px; padding: 18px 24px;
    display: flex; align-items: center; justify-content: space-between;
    margin: 1.2rem 0; box-shadow: 0 4px 18px rgba(0,0,0,0.12);
}
.score-banner.balanced {
    background: linear-gradient(120deg, #1a4d1a 0%, #2d7a1b 100%);
}
.score-banner.not-balanced {
    background: linear-gradient(120deg, #7a1a1a 0%, #b83232 100%);
}
.score-banner h2 {
    font-family: 'DM Serif Display', serif;
    color: white; margin: 0; font-size: 1.6rem;
}
.score-banner p { color: rgba(255,255,255,0.8); margin: 2px 0 0; font-size: 0.85rem; }
.score-badge {
    background: rgba(255,255,255,0.2); border-radius: 50px;
    padding: 8px 20px; color: white; font-size: 1.1rem; font-weight: 600;
}

/* ── Metric Cards ── */
.metric-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 0.8rem 0; }
.metric-card {
    background: white; border: 1.5px solid #e8e8e8;
    border-radius: 12px; padding: 14px 16px; text-align: center;
}
.metric-card .mc-label { font-size: 0.78rem; color: #888; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-card .mc-value { font-family: 'DM Serif Display', serif; font-size: 1.8rem; color: #1a1a1a; line-height: 1.1; margin: 4px 0; }
.metric-card .mc-sub { font-size: 0.78rem; color: #999; }

/* ── Section Headers ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.25rem; color: #1a3a0a; margin: 1.4rem 0 0.7rem;
    padding-bottom: 6px; border-bottom: 2px solid #d4e8c2;
}

/* ── Minuman Info Box ── */
.minuman-box {
    background: linear-gradient(135deg, #e8f4fd 0%, #d6ecf9 100%);
    border: 1.5px solid #90caf9; border-left: 4px solid #1976d2;
    border-radius: 12px; padding: 14px 18px; margin: 0.8rem 0;
}
.minuman-box h4 { color: #0d47a1; margin: 0 0 6px; font-size: 0.95rem; }
.minuman-box p  { color: #1a4a7a; margin: 0; font-size: 0.83rem; line-height: 1.5; }
.minuman-badge {
    display: inline-block; background: #1976d2; color: white;
    border-radius: 20px; padding: 2px 10px; font-size: 0.78rem;
    margin: 2px 3px; font-weight: 500;
}

/* ── Missing / Warning ── */
.missing-box {
    background: #fff8e1; border: 1.5px solid #ffcc02;
    border-left: 4px solid #f59e0b; border-radius: 10px;
    padding: 12px 16px; font-size: 0.84rem; color: #5a3e00;
}
.missing-box strong { color: #3a2500; }

/* ── Error Box ── */
.err-box {
    background: #fdf0f0; border: 1.5px solid #f5c2c2;
    border-left: 4px solid #e53935; border-radius: 10px;
    padding: 14px 18px; font-size: 0.85rem; color: #4a1010;
}

/* ── AKG Row ── */
.akg-row { display: flex; align-items: center; gap: 10px; margin: 6px 0; }
.akg-label { width: 110px; font-size: 0.85rem; font-weight: 500; color: #333; flex-shrink: 0; }
.akg-detail { font-size: 0.78rem; color: #888; width: 110px; flex-shrink: 0; text-align: right; }
.akg-status {
    font-size: 0.75rem; font-weight: 700; padding: 2px 10px;
    border-radius: 20px; white-space: nowrap;
}
.akg-kurang  { background: #fde8e8; color: #c0392b; }
.akg-cukup   { background: #e8f5e9; color: #27ae60; }
.akg-berlebih{ background: #fff3e0; color: #e67e22; }

/* ── Sidebar ── */
.sidebar-section { margin: 0.8rem 0; }
.sidebar-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600; margin-bottom: 4px; }
.sidebar-ref { font-size: 0.73rem; color: #aaa; line-height: 1.6; }

/* ── Footer ── */
.sp-footer {
    text-align: center; padding: 1.5rem; margin-top: 2rem;
    border-top: 1px solid #e8e8e8; color: #aaa; font-size: 0.8rem; line-height: 1.8;
}
.sp-footer strong { color: #555; }

/* ── Streamlit overrides ── */
div[data-testid="stExpander"] { border: 1.5px solid #d4e8c2 !important; border-radius: 10px !important; }
.stButton > button {
    background: linear-gradient(135deg, #2d5a1b 0%, #3d7a25 100%) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    font-size: 1rem !important; padding: 0.6rem 0 !important;
    box-shadow: 0 4px 14px rgba(45,90,27,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 18px rgba(45,90,27,0.4) !important; }
[data-testid="stFileUploader"] { border: 2px dashed #c4ddb0 !important; border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# CORE FUNCTIONS
# ==============================================================================

def resize_image_for_inference(image, max_size=1280):
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def estimate_weight_from_mask(mask, class_name, pixel_to_cm):
    """
    Estimasi berat dari mask segmentasi.
    Formula: Berat (g) = Area_2D (cm²) × Tinggi_asumsi (cm) × Densitas (g/cm³)
    Referensi: Fang et al. (2011); Pouladzadeh et al. (2014)
    """
    HEIGHT_CM = {'buah': 3.5, 'karbohidrat': 2.5, 'minuman': 10.0, 'protein': 3.0, 'sayur': 2.0}
    area_pixels = np.sum(mask > 0)
    area_cm2    = area_pixels * (pixel_to_cm ** 2)
    volume_cm3  = area_cm2 * HEIGHT_CM.get(class_name, 2.5)
    return volume_cm3 * NUTRITION_DB[class_name]['density']


def calculate_nutrition_from_grams(class_name, weight_grams):
    db     = NUTRITION_DB[class_name]
    factor = weight_grams / 100
    return {
        'kalori':       db['kalori_per_100g']       * factor,
        'protein':      db['protein_per_100g']      * factor,
        'karbohidrat':  db['karbohidrat_per_100g']  * factor,
        'lemak':        db['lemak_per_100g']         * factor,
        'serat':        db['serat_per_100g']         * factor,
    }


def detect_plate_circle(image_np):
    """
    Hough Circle Transform untuk kalibrasi piring.
    Asumsi diameter standar piring = 25 cm.
    Referensi: Ballard (1981); Puri et al. (2009)
    """
    gray    = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=50, param2=30, minRadius=50, maxRadius=500
    )
    if circles is not None:
        c = max(np.round(circles[0]).astype(int), key=lambda x: x[2])
        return 25.0 / (c[2] * 2), True
    return 0.05, False


@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("⏳ Mengunduh model YOLOv8… (~12 MB)"):
                gdown.download(f'https://drive.google.com/uc?id={MODEL_ID}', MODEL_PATH, quiet=False)
        from ultralytics import YOLO
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None


def process_segmentation_results(image, results, conf_threshold):
    img_array    = np.array(image)
    annotated    = img_array.copy()
    pixel_to_cm, plate_detected = detect_plate_circle(img_array)

    COLORS = {
        'buah':        (232,  84,  74),
        'karbohidrat': (245, 166,  35),
        'minuman':     ( 74, 144, 217),
        'protein':     (192,  57,  43),
        'sayur':       ( 39, 174,  96),
    }

    food_detections    = []   # buah, karbo, protein, sayur
    minuman_detections = []   # minuman only

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.data.cpu().numpy()

        for mask, box in zip(masks, boxes):
            conf       = float(box[4])
            class_id   = int(box[5])
            class_name = CLASSES[class_id]
            if conf < conf_threshold:
                continue

            mask_rsz = cv2.resize(
                mask, (img_array.shape[1], img_array.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            weight    = estimate_weight_from_mask(mask_rsz, class_name, pixel_to_cm)
            nutrition = calculate_nutrition_from_grams(class_name, weight)

            record = {
                'class': class_name,
                'weight_grams': weight,
                'confidence': conf,
                **nutrition
            }

            if class_name == 'minuman':
                minuman_detections.append(record)
            else:
                food_detections.append(record)

            # Draw overlay
            color   = COLORS.get(class_name, (128, 128, 128))
            overlay = np.zeros_like(img_array)
            overlay[mask_rsz > 0.5] = color
            annotated = cv2.addWeighted(annotated, 1, overlay, 0.4, 0)

            x1, y1, x2, y2 = map(int, box[:4])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            lbl = f"{NUTRITION_DB[class_name]['emoji']} {class_name} {weight:.0f}g ({conf:.2f})"
            cv2.putText(annotated, lbl, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated, food_detections, minuman_detections, pixel_to_cm, plate_detected


def analyze_nutrition_balance(food_detections, user_type):
    """
    Analisis hanya dari food_detections (buah, karbo, protein, sayur).
    Minuman TIDAK dimasukkan ke kalkulasi ini.
    """
    if not food_detections:
        return None

    total_nutrition = {
        'kalori':      sum(d['kalori']      for d in food_detections),
        'protein':     sum(d['protein']     for d in food_detections),
        'karbohidrat': sum(d['karbohidrat'] for d in food_detections),
        'lemak':       sum(d['lemak']       for d in food_detections),
        'serat':       sum(d['serat']       for d in food_detections),
    }

    # Berat per kategori makanan
    cat_weights = {c: sum(d['weight_grams'] for d in food_detections if d['class'] == c)
                   for c in FOOD_CLASSES}
    total_food_weight = sum(cat_weights.values())

    composition = {c: (cat_weights[c] / total_food_weight * 100) if total_food_weight > 0 else 0
                   for c in FOOD_CLASSES}

    # AKG comparison
    akg = AKG_DATABASE[user_type]
    akg_comparison = {}
    for nutrient in ['kalori', 'protein', 'karbohidrat', 'lemak', 'serat']:
        pct = (total_nutrition[nutrient] / akg[nutrient]) * 100
        if pct < 80:
            status, color, css = 'KURANG',   '#c0392b', 'akg-kurang'
        elif pct <= 120:
            status, color, css = 'CUKUP',    '#27ae60', 'akg-cukup'
        else:
            status, color, css = 'BERLEBIH', '#e67e22', 'akg-berlebih'
        akg_comparison[nutrient] = {
            'percentage': pct, 'status': status,
            'color': color, 'css': css,
            'target': akg[nutrient], 'actual': total_nutrition[nutrient]
        }

    # Balance Score
    detected_food_cats = sum(1 for c in FOOD_CLASSES if cat_weights[c] > 0)
    completeness_score = (detected_food_cats / 4) * 60

    comp_deviation = sum(abs(composition[c] - IDEAL_COMPOSITION[c]) for c in FOOD_CLASSES)
    composition_score = max(0, 40 - (comp_deviation / (100 * 4)) * 40)

    balance_score = completeness_score + composition_score

    missing = [NUTRITION_DB[c]['name'] for c in FOOD_CLASSES if cat_weights[c] == 0]

    return {
        'total_nutrition':    total_nutrition,
        'cat_weights':        cat_weights,
        'composition':        composition,
        'akg_comparison':     akg_comparison,
        'balance_score':      balance_score,
        'completeness_score': completeness_score,
        'composition_score':  composition_score,
        'missing_categories': missing,
        'detected_food_cats': detected_food_cats,
        'total_food_weight':  total_food_weight,
    }


# ==============================================================================
# DISPLAY FUNCTIONS
# ==============================================================================

def show_header():
    st.markdown("""
    <div class="sp-header">
        <div class="sp-header-icon">🍽️</div>
        <div class="sp-header-text">
            <h1>SmartPlate</h1>
            <p>ANALISIS NUTRISI & KESEIMBANGAN GIZI · POWERED BY YOLOV8 SEGMENTATION</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def show_guide():
    with st.expander("📋 Panduan Penggunaan", expanded=False):
        st.markdown("""
        <div class="guide-grid">
            <div class="guide-card">
                <div class="guide-card-icon">📸</div>
                <div class="guide-card-text">
                    <strong>Satu Foto, Semua Makanan</strong>
                    Pastikan seluruh makanan di piring terlihat jelas dalam satu gambar.
                </div>
            </div>
            <div class="guide-card">
                <div class="guide-card-icon">🍽️</div>
                <div class="guide-card-text">
                    <strong>Piring Putih Ø 25 cm</strong>
                    Gunakan piring putih polos untuk kalibrasi ukuran porsi yang lebih akurat.
                </div>
            </div>
            <div class="guide-card">
                <div class="guide-card-icon">📐</div>
                <div class="guide-card-text">
                    <strong>Foto dari Atas (Top-view)</strong>
                    Jarak ideal 30–50 cm, cahaya cukup, tanpa bayangan yang menghalangi.
                </div>
            </div>
            <div class="guide-card">
                <div class="guide-card-icon">✋</div>
                <div class="guide-card-text">
                    <strong>Minimalisir Overlap</strong>
                    Susun makanan agar tidak saling menutupi satu sama lain.
                </div>
            </div>
        </div>
        <div class="disclaimer-card">
            <strong>⚠️ Tentang Akurasi Sistem</strong><br>
            Estimasi berat menggunakan formula <em>Area × Tinggi × Densitas</em> dari mask segmentasi 2D, 
            dengan akurasi ±15–30% dari berat aktual. Nilai nutrisi diambil dari database rata-rata per kategori 
            (FatSecret Indonesia), bukan dari jenis makanan spesifik. Minuman dideteksi namun 
            <strong>tidak dihitung</strong> dalam analisis kalori & Isi Piringku — sesuai dengan fokus 
            estimasi makanan di atas piring. Sistem ini merupakan <em>baseline research</em>; 
            untuk kebutuhan medis, konsultasikan dengan ahli gizi.
        </div>
        """, unsafe_allow_html=True)


def show_minuman_info(minuman_detections):
    """Tampilkan minuman sebagai informasi, bukan kalkulasi."""
    if not minuman_detections:
        return

    badges = ''.join(
        f'<span class="minuman-badge">🥤 minuman ({d["confidence"]:.0%})</span>'
        for d in minuman_detections
    )
    note = (
        f"{len(minuman_detections)} objek minuman terdeteksi. "
        "Minuman <strong>tidak dihitung</strong> dalam estimasi kalori dan analisis "
        "Isi Piringku karena fokus sistem adalah makanan di atas piring. "
        "Air putih (0 kkal) dan minuman lain dianggap sebagai pelengkap konsumsi sehari-hari."
    )
    st.markdown(f"""
    <div class="minuman-box">
        <h4>🥤 Minuman Terdeteksi (Informasi)</h4>
        <div style="margin-bottom:6px;">{badges}</div>
        <p>{note}</p>
    </div>
    """, unsafe_allow_html=True)


def show_missing_warning(missing):
    if not missing:
        return
    items = ' · '.join(f'<strong>{m}</strong>' for m in missing)
    st.markdown(f"""
    <div class="missing-box">
        ⚠️ Kategori tidak terdeteksi: {items}<br>
        <span style="font-size:0.8rem;">Jika kategori tersebut ada di piring Anda, coba foto ulang dengan pencahayaan lebih baik dan pastikan makanan tidak tertutup.</span>
    </div>
    """, unsafe_allow_html=True)


def show_balance_score(analysis):
    score = analysis['balance_score']
    is_ok = score >= 70
    css   = 'balanced' if is_ok else 'not-balanced'
    emoji = '✅' if is_ok else '⚠️'
    label = 'Seimbang' if is_ok else 'Kurang Seimbang'
    cats  = f"{analysis['detected_food_cats']}/4 kategori makanan terdeteksi"

    st.markdown(f"""
    <div class="score-banner {css}">
        <div>
            <h2>{emoji} {label}</h2>
            <p>{cats}</p>
        </div>
        <div class="score-badge">{score:.1f} / 100</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Completeness</div>
            <div class="mc-value">{analysis['completeness_score']:.0f}<span style="font-size:1rem;color:#aaa">/60</span></div>
            <div class="mc-sub">Kelengkapan kategori makanan</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">Composition</div>
            <div class="mc-value">{analysis['composition_score']:.0f}<span style="font-size:1rem;color:#aaa">/40</span></div>
            <div class="mc-sub">Kesesuaian proporsi Isi Piringku</div>
        </div>
        """, unsafe_allow_html=True)


def show_food_detail_table(food_detections):
    st.markdown('<div class="section-title">📋 Detail Makanan Terdeteksi</div>', unsafe_allow_html=True)
    rows = []
    for i, d in enumerate(food_detections, 1):
        rows.append({
            'No': i,
            'Kategori': f"{NUTRITION_DB[d['class']]['emoji']} {d['class']}",
            'Berat (g)':     f"{d['weight_grams']:.1f}",
            'Kalori (kkal)': f"{d['kalori']:.1f}",
            'Protein (g)':   f"{d['protein']:.1f}",
            'Karbo (g)':     f"{d['karbohidrat']:.1f}",
            'Lemak (g)':     f"{d['lemak']:.1f}",
            'Serat (g)':     f"{d['serat']:.1f}",
            'Conf.':         f"{d['confidence']:.0%}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def show_akg_comparison(analysis):
    st.markdown('<div class="section-title">📊 Kecukupan Gizi vs AKG 2019</div>', unsafe_allow_html=True)
    st.caption("KURANG < 80% · CUKUP 80–120% · BERLEBIH > 120% — Permenkes No. 28 Tahun 2019")

    labels_map = {
        'kalori': 'Kalori', 'protein': 'Protein',
        'karbohidrat': 'Karbohidrat', 'lemak': 'Lemak', 'serat': 'Serat'
    }
    units_map = {
        'kalori': 'kkal', 'protein': 'g',
        'karbohidrat': 'g', 'lemak': 'g', 'serat': 'g'
    }

    for nutrient, data in analysis['akg_comparison'].items():
        unit = units_map[nutrient]
        c1, c2, c3, c4 = st.columns([1.6, 3, 1.4, 1])
        with c1:
            st.markdown(f"**{labels_map[nutrient]}**")
        with c2:
            st.progress(min(data['percentage'] / 100, 1.0))
        with c3:
            st.markdown(
                f"<span style='font-size:0.8rem;color:#888'>"
                f"{data['actual']:.1f} / {data['target']:.0f} {unit}</span>",
                unsafe_allow_html=True
            )
        with c4:
            st.markdown(
                f"<span class='akg-status {data['css']}'>{data['status']}</span>",
                unsafe_allow_html=True
            )


def show_composition_charts(analysis):
    st.markdown('<div class="section-title">🥧 Komposisi Piring vs Isi Piringku</div>', unsafe_allow_html=True)

    CHART_COLORS = {
        'buah':        '#E8544A',
        'karbohidrat': '#F5A623',
        'protein':     '#C0392B',
        'sayur':       '#27AE60',
    }

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
        labels, sizes, colors_pie = [], [], []
        for c in FOOD_CLASSES:
            pct = analysis['composition'][c]
            if pct > 0.5:
                labels.append(f"{NUTRITION_DB[c]['emoji']} {NUTRITION_DB[c]['name']}\n{pct:.1f}%")
                sizes.append(pct)
                colors_pie.append(CHART_COLORS[c])
        if sizes:
            wedges, texts, autotexts = ax.pie(
                sizes, labels=labels, colors=colors_pie,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 9},
                wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
            )
            for at in autotexts:
                at.set_fontsize(8)
        ax.set_title('Aktual', fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(5, 5), facecolor='none')
        ideal_labels  = [f"{NUTRITION_DB[c]['emoji']} {NUTRITION_DB[c]['name']}\n{IDEAL_COMPOSITION[c]}%" for c in FOOD_CLASSES]
        ideal_sizes   = [IDEAL_COMPOSITION[c] for c in FOOD_CLASSES]
        ideal_colors  = [CHART_COLORS[c] for c in FOOD_CLASSES]
        ax.pie(
            ideal_sizes, labels=ideal_labels, colors=ideal_colors,
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 9},
            wedgeprops={'linewidth': 1.5, 'edgecolor': 'white'}
        )
        ax.set_title('Ideal (Isi Piringku)', fontsize=12, fontweight='bold', pad=10)
        st.pyplot(fig)
        plt.close(fig)

    st.caption("📌 Isi Piringku — Kemenkes RI (2017) · Minuman tidak termasuk proporsi piring")


def display_full_analysis(analysis, food_detections, minuman_detections):
    """Orchestrate the full result display."""
    show_minuman_info(minuman_detections)
    show_missing_warning(analysis['missing_categories'])
    show_balance_score(analysis)

    st.divider()
    show_food_detail_table(food_detections)
    st.divider()
    show_akg_comparison(analysis)
    st.divider()
    show_composition_charts(analysis)


# ==============================================================================
# SIDEBAR
# ==============================================================================

def build_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0 0.5rem;">
            <div style="font-size:2.2rem;">🍽️</div>
            <div style="font-family:'DM Serif Display',serif;font-size:1.3rem;color:#1a3a0a;font-weight:bold;">SmartPlate</div>
            <div style="font-size:0.73rem;color:#888;letter-spacing:0.5px;">NUTRITION BALANCE DETECTOR</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("**⚙️ Pengaturan Deteksi**")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, CONFIDENCE_THRESHOLD, 0.05,
                                   help="Ambang batas kepercayaan deteksi. Semakin tinggi = semakin selektif.")

        user_type = st.selectbox(
            "Profil AKG",
            list(AKG_DATABASE.keys()),
            format_func=lambda x: AKG_DATABASE[x]['label'],
            help="Profil angka kecukupan gizi harian sesuai Permenkes No. 28/2019"
        )

        st.divider()

        akg = AKG_DATABASE[user_type]
        st.markdown('<div class="sidebar-label">Target AKG Harian</div>', unsafe_allow_html=True)
        metrics = [
            ("Kalori", f"{akg['kalori']} kkal"),
            ("Protein", f"{akg['protein']} g"),
            ("Karbohidrat", f"{akg['karbohidrat']} g"),
            ("Lemak", f"{akg['lemak']} g"),
            ("Serat", f"{akg['serat']} g"),
        ]
        for label, val in metrics:
            c1, c2 = st.columns(2)
            c1.caption(label)
            c2.markdown(f"**{val}**")

        st.divider()

        st.markdown('<div class="sidebar-label">Komposisi Ideal Isi Piringku</div>', unsafe_allow_html=True)
        for cls, pct in IDEAL_COMPOSITION.items():
            info = NUTRITION_DB[cls]
            st.write(f"{info['emoji']} **{info['name']}** — {pct}%")
        st.caption("*Minuman: dideteksi sebagai informasi, tidak dihitung dalam proporsi piring")

        st.divider()

        st.markdown('<div class="sidebar-label">Kategori Deteksi</div>', unsafe_allow_html=True)
        for cls in CLASSES:
            tag = " *(info)*" if cls == 'minuman' else ""
            st.write(f"{NUTRITION_DB[cls]['emoji']} {NUTRITION_DB[cls]['name']}{tag}")

        st.divider()

        st.markdown('<div class="sidebar-label">Referensi Ilmiah</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="sidebar-ref">
        FatSecret Indonesia · FAO/INFOODS (2012) · Antarlina et al. (2009) ·
        Permenkes 28/2019 · Kemenkes RI (2017) · Fang et al. (2011) ·
        Pouladzadeh et al. (2014) · Ballard (1981)
        </div>
        """, unsafe_allow_html=True)

    return conf_threshold, user_type


# ==============================================================================
# MAIN APP
# ==============================================================================

def run_analysis(image, conf_threshold, user_type, model):
    image_rsz = resize_image_for_inference(image, max_size=1280)
    w, h      = image.size

    results = model.predict(
        source=image_rsz, conf=conf_threshold,
        iou=IOU_THRESHOLD, verbose=False
    )

    annotated, food_dets, minuman_dets, pixel_to_cm, plate_ok = \
        process_segmentation_results(image_rsz, results, conf_threshold)

    analysis = analyze_nutrition_balance(food_dets, user_type)

    return {
        'annotated':       annotated,
        'food_dets':       food_dets,
        'minuman_dets':    minuman_dets,
        'pixel_to_cm':     pixel_to_cm,
        'plate_ok':        plate_ok,
        'analysis':        analysis,
        'original_size':   (w, h),
        'resized':         max(w, h) > 1280,
    }


def show_calibration_info(result):
    if result['plate_ok']:
        st.success(f"✅ Piring terdeteksi — kalibrasi: {result['pixel_to_cm']:.4f} cm/pixel")
    else:
        st.warning(f"⚠️ Piring tidak terdeteksi — fallback calibration: {result['pixel_to_cm']:.4f} cm/pixel")
        st.caption("💡 Gunakan piring putih polos Ø 25 cm untuk estimasi yang lebih akurat.")


def main():
    show_header()
    conf_threshold, user_type = build_sidebar()
    show_guide()

    model = load_model()
    if model is None:
        st.markdown("""
        <div class="err-box">
            <strong>❌ Model tidak berhasil dimuat.</strong><br>
            Pastikan MODEL_ID sudah benar dan file di Google Drive bersifat <em>Anyone with the link can view</em>.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("")

    tab_upload, tab_camera = st.tabs(["📤  Upload Gambar", "📸  Kamera"])

    # ── TAB UPLOAD ────────────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Pilih foto piring makanan Anda",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded)
            col_orig, col_result = st.columns(2)

            with col_orig:
                st.image(image, caption="Foto Asli", use_container_width=True)
                w, h = image.size
                if max(w, h) > 1280:
                    st.caption(f"📱 Resolusi {w}×{h}px — akan dioptimasi otomatis.")

            if st.button("🔍  Analisis Nutrisi", type="primary", use_container_width=True):
                with st.spinner("Menganalisis komposisi makanan Anda…"):
                    result = run_analysis(image, conf_threshold, user_type, model)
                    st.session_state['result'] = result

            if 'result' in st.session_state:
                res = st.session_state['result']
                with col_result:
                    st.image(res['annotated'], caption="Hasil Segmentasi", use_container_width=True)

                show_calibration_info(res)
                st.divider()

                if res['food_dets']:
                    display_full_analysis(res['analysis'], res['food_dets'], res['minuman_dets'])
                elif res['minuman_dets']:
                    show_minuman_info(res['minuman_dets'])
                    st.markdown("""
                    <div class="err-box">
                        <strong>🥤 Hanya minuman yang terdeteksi.</strong><br>
                        Tidak ada makanan (buah, karbohidrat, protein, sayur) yang ditemukan dalam foto.
                        Pastikan piring berisi makanan dan terlihat jelas dari atas.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="err-box">
                        <strong>❌ Tidak ada makanan atau minuman yang terdeteksi.</strong><br>
                        Kemungkinan penyebab: pencahayaan kurang, foto terlalu jauh, makanan terlalu kecil, 
                        atau gambar tidak fokus. Coba foto ulang dari jarak 30–50 cm dengan cahaya yang cukup.
                    </div>
                    """, unsafe_allow_html=True)

    # ── TAB CAMERA ───────────────────────────────────────────────────────────
    with tab_camera:
        camera_img = st.camera_input("Arahkan kamera ke piring makanan Anda", label_visibility="collapsed")

        if camera_img:
            image = Image.open(camera_img)
            with st.spinner("Menganalisis foto dari kamera…"):
                result = run_analysis(image, conf_threshold, user_type, model)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.image(image, caption="Foto Kamera", use_container_width=True)
            with col_r2:
                st.image(result['annotated'], caption="Hasil Segmentasi", use_container_width=True)

            show_calibration_info(result)
            st.divider()

            if result['food_dets']:
                display_full_analysis(result['analysis'], result['food_dets'], result['minuman_dets'])
            elif result['minuman_dets']:
                show_minuman_info(result['minuman_dets'])
                st.warning("Tidak ada makanan yang terdeteksi, hanya minuman.")
            else:
                st.markdown("""
                <div class="err-box">
                    <strong>❌ Tidak ada makanan yang terdeteksi.</strong> Coba foto ulang dengan pencahayaan lebih baik.
                </div>
                """, unsafe_allow_html=True)

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sp-footer">
        <strong>SmartPlate</strong> — Nutrition Balance Detector<br>
        © 2026 Mochamad Faisal Akbar · YOLOv8 Instance Segmentation<br>
        <span style="font-size:0.73rem;">
        FatSecret Indonesia · FAO/INFOODS (2012) · Antarlina et al. (2009) · Permenkes 28/2019 ·
        Isi Piringku – Kemenkes RI (2017) · Fang et al. (2011) · Pouladzadeh et al. (2014) · Ballard (1981)
        </span><br>
        <em style="font-size:0.73rem;">Untuk kebutuhan medis atau diet klinis, konsultasikan dengan ahli gizi profesional.</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
