import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import gdown
import matplotlib.pyplot as plt
import pandas as pd

# ⭐ MUST BE FIRST - Only ONE st.set_page_config()
st.set_page_config(
    page_title="SmartPlate - Nutrition Analysis",
    page_icon="🍽️",
    layout="wide"
)

#==============================================================================
# CONFIGURATION
#==============================================================================

MODEL_ID = '1KPbuf5rjNLT9oRsQuZ8f3Xncl4qoIGBL'  # ⚠️ CHANGE THIS to your model ID
MODEL_PATH = 'best_nutrition_segmentation.pt'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASSES = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']

#==============================================================================
# NUTRITION DATABASE
#==============================================================================

NUTRITION_DB = {
    'buah': {
        'name': 'Buah',
        'emoji': '🍎',
        'density': 0.8,
        'kalori_per_100g': 52,
        'protein_per_100g': 0.3,
        'karbohidrat_per_100g': 14,
        'lemak_per_100g': 0.2,
        'serat_per_100g': 2.4
    },
    'karbohidrat': {
        'name': 'Karbohidrat',
        'emoji': '🍚',
        'density': 1.0,
        'kalori_per_100g': 130,
        'protein_per_100g': 2.7,
        'karbohidrat_per_100g': 28,
        'lemak_per_100g': 0.3,
        'serat_per_100g': 0.4
    },
    'minuman': {
        'name': 'Minuman',
        'emoji': '🥤',
        'density': 1.0,
        'kalori_per_100g': 42,
        'protein_per_100g': 0,
        'karbohidrat_per_100g': 11,
        'lemak_per_100g': 0,
        'serat_per_100g': 0
    },
    'protein': {
        'name': 'Protein',
        'emoji': '🍗',
        'density': 1.1,
        'kalori_per_100g': 165,
        'protein_per_100g': 31,
        'karbohidrat_per_100g': 0,
        'lemak_per_100g': 3.6,
        'serat_per_100g': 0
    },
    'sayur': {
        'name': 'Sayur',
        'emoji': '🥗',
        'density': 0.6,
        'kalori_per_100g': 23,
        'protein_per_100g': 2.9,
        'karbohidrat_per_100g': 3.6,
        'lemak_per_100g': 0.4,
        'serat_per_100g': 2.6
    }
}

IDEAL_COMPOSITION = {
    'buah': {'percentage': 15},
    'karbohidrat': {'percentage': 30},
    'minuman': {'percentage': 10},
    'protein': {'percentage': 20},
    'sayur': {'percentage': 25}
}

AKG_DATABASE = {
    'male_adult': {
        'kalori': 2150, 'protein': 62, 'karbohidrat': 340,
        'lemak': 67, 'serat': 37
    },
    'female_adult': {
        'kalori': 1900, 'protein': 56, 'karbohidrat': 300,
        'lemak': 59, 'serat': 32
    },
    'child': {
        'kalori': 1650, 'protein': 45, 'karbohidrat': 250,
        'lemak': 50, 'serat': 25
    },
    'pregnant_trimester1': {
        'kalori': 2080, 'protein': 57, 'karbohidrat': 325,
        'lemak': 61.3, 'serat': 35
    },
    'pregnant_trimester2': {
        'kalori': 2200, 'protein': 66, 'karbohidrat': 340,
        'lemak': 61.3, 'serat': 36
    },
    'pregnant_trimester3': {
        'kalori': 2200, 'protein': 86, 'karbohidrat': 340,
        'lemak': 61.3, 'serat': 36
    }
}

#==============================================================================
# CUSTOM CSS
#==============================================================================

st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {
        width: 100%; 
        background-color: #0245d6; 
        color: white;
        border-radius: 10px;
        height: 50px;
        font-size: 18px;
        font-weight: bold;
    }
    .balance-indicator {
        padding: 20px; 
        border-radius: 10px; 
        text-align: center;
        font-size: 24px; 
        font-weight: bold; 
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .balanced {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        color: white;
    }
    .not-balanced {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
        color: white;
    }
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .warning-box {
        background-color: #fff4e5;
        border: 2px solid #ff9800;
        border-left: 6px solid #f57c00;
        padding: 18px;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(245, 124, 0, 0.15);
    }
    .info-box {
        background-color: #e3f2fd;
        border: 2px solid #42a5f5;
        border-left: 6px solid #1976d2;
        padding: 18px;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(25, 118, 210, 0.15);
    }
    .error-box {
        background-color: #ffebee;
        border: 2px solid #ef5350;
        border-left: 6px solid #d32f2f;
        padding: 18px;
        margin: 20px 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(211, 47, 47, 0.15);
    }
    </style>
""", unsafe_allow_html=True)

#==============================================================================
# HELPER FUNCTIONS
#==============================================================================

def calculate_nutrition_from_grams(class_name, weight_grams):
    """Calculate nutrition from weight"""
    db = NUTRITION_DB[class_name]
    factor = weight_grams / 100
    
    return {
        'kalori': db['kalori_per_100g'] * factor,
        'protein': db['protein_per_100g'] * factor,
        'karbohidrat': db['karbohidrat_per_100g'] * factor,
        'lemak': db['lemak_per_100g'] * factor,
        'serat': db['serat_per_100g'] * factor
    }

def calculate_percentage_of_akg(nutrition, user_type='male_adult'):
    """Calculate percentage of daily recommended intake"""
    akg = AKG_DATABASE[user_type]
    return {
        key: (nutrition[key] / akg[key] * 100) if key in akg else 0
        for key in nutrition
    }

def detect_plate_and_calibrate(image):
    """Detect circular plate for calibration"""
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
        param1=50, param2=30, minRadius=150, maxRadius=500
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        largest_circle = circles[0][np.argmax(circles[0, :, 2])]
        radius_px = largest_circle[2]
        
        # Assume standard 25cm diameter plate
        diameter_cm = 25
        pixel_to_cm = diameter_cm / (2 * radius_px)
        
        return pixel_to_cm, largest_circle
    
    # Fallback calibration if no plate detected
    return 0.05, None

def calculate_weight_from_mask(mask_array, pixel_to_cm, class_name):
    """
    Convert segmentation mask to weight (grams)
    
    Formula: Berat = Area_2D × Tinggi_rata-rata × Densitas
    
    CATATAN: Ini adalah PERKIRAAN berdasarkan asumsi:
    - Tinggi makanan diasumsikan konstan per kategori
    - Densitas diasumsikan konstan per kategori
    - Estimasi dapat berbeda 15-30% dari berat aktual
    """
    mask_pixels = np.sum(mask_array > 0)
    area_cm2 = mask_pixels * (pixel_to_cm ** 2)
    
    # Asumsi tinggi rata-rata per kategori (dalam cm)
    # Berdasarkan observasi porsi normal makanan Indonesia
    avg_heights = {
        'karbohidrat': 2.5,   # Tumpukan nasi/roti
        'protein': 3.0,       # Ketebalan ayam/ikan
        'sayur': 2.0,         # Tumpukan sayur tumis/kuah
        'buah': 3.5,          # Ketebalan potongan buah
        'minuman': 10.0       # Tinggi cairan dalam gelas
    }
    height_cm = avg_heights.get(class_name, 2.5)
    
    volume_cm3 = area_cm2 * height_cm
    density = NUTRITION_DB[class_name]['density']
    weight_grams = volume_cm3 * density
    
    return {
        'mask_pixels': int(mask_pixels),
        'area_cm2': float(area_cm2),
        'volume_cm3': float(volume_cm3),
        'weight_grams': float(weight_grams),
        'estimated_height_cm': height_cm
    }

#==============================================================================
# MODEL LOADING
#==============================================================================

@st.cache_resource
def load_model():
    """Load YOLOv8 segmentation model"""
    try:
        from ultralytics import YOLO
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('📥 Downloading model from Google Drive...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success('✅ Model downloaded!')
        
        with st.spinner('🔄 Loading YOLOv8 segmentation model...'):
            model = YOLO(MODEL_PATH)
            st.success('✅ Model loaded successfully!')
        
        return model
    
    except ImportError:
        st.error("❌ Ultralytics not installed. Run: pip install ultralytics")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

#==============================================================================
# RESULT PROCESSING
#==============================================================================

def process_segmentation_results(image, results, conf_threshold=0.25):
    """Process YOLOv8 segmentation results"""
    pixel_to_cm, plate_circle = detect_plate_and_calibrate(image)
    
    img_array = np.array(image).copy()
    overlay = img_array.copy()
    detections = []
    
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        return Image.fromarray(img_array), [], None, False
    
    for i, (box, mask, cls, conf) in enumerate(zip(
        result.boxes.xyxy, result.masks.data, result.boxes.cls, result.boxes.conf
    )):
        if conf < conf_threshold:
            continue
        
        class_idx = int(cls)
        class_name = CLASSES[class_idx]
        confidence = float(conf)
        
        mask_array = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
        
        weight_info = calculate_weight_from_mask(mask_resized, pixel_to_cm, class_name)
        weight_grams = weight_info['weight_grams']
        nutrition = calculate_nutrition_from_grams(class_name, weight_grams)
        
        color = plt.cm.tab10(class_idx)[:3]
        color_rgb = tuple([int(c * 255) for c in color])
        
        mask_bool = mask_resized > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * 0.6 + color_rgb[c] * 0.4,
                overlay[:, :, c]
            )
        
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color_rgb, 2)
        
        label = f"{class_name}: {weight_grams:.0f}g"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(img_array, (x1, y1 - text_h - 10), 
                     (x1 + text_w + 10, y1), color_rgb, -1)
        cv2.putText(img_array, label, (x1 + 5, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
        
        detections.append({
            'id': i + 1,
            'class': class_name,
            'confidence': confidence,
            'weight_grams': weight_grams,
            'nutrition': nutrition,
            'weight_info': weight_info
        })
    
    final_img = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    if plate_circle is not None:
        cx, cy, r = plate_circle
        cv2.circle(final_img, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(final_img, f"Plate: {r*2*pixel_to_cm:.1f}cm", 
                   (cx-50, cy-r-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
    
    plate_detected = plate_circle is not None
    
    return Image.fromarray(final_img), detections, pixel_to_cm, plate_detected

#==============================================================================
# NUTRITION ANALYSIS
#==============================================================================

def analyze_nutrition_balance(detections, user_type='male_adult'):
    """Comprehensive nutrition analysis"""
    
    if not detections:
        return None
    
    total_nutrition = {
        'kalori': 0, 'protein': 0, 'lemak': 0,
        'karbohidrat': 0, 'serat': 0
    }
    
    composition = {}
    total_weight = 0
    
    for det in detections:
        class_name = det['class']
        weight = det['weight_grams']
        nutrition = det['nutrition']
        
        for key in total_nutrition:
            if key in nutrition:
                total_nutrition[key] += nutrition[key]
        
        if class_name not in composition:
            composition[class_name] = {'weight': 0, 'count': 0}
        composition[class_name]['weight'] += weight
        composition[class_name]['count'] += 1
        total_weight += weight
    
    for class_name in composition:
        composition[class_name]['percentage'] = \
            (composition[class_name]['weight'] / total_weight * 100) if total_weight > 0 else 0
    
    detected_classes = set(composition.keys())
    required_classes = set(CLASSES)
    missing_classes = required_classes - detected_classes
    completeness_score = (len(detected_classes) / len(required_classes)) * 100
    
    composition_deviations = {}
    total_deviation = 0
    
    for class_name in CLASSES:
        ideal_pct = IDEAL_COMPOSITION[class_name]['percentage']
        actual_pct = composition.get(class_name, {}).get('percentage', 0)
        deviation = actual_pct - ideal_pct
        composition_deviations[class_name] = deviation
        total_deviation += abs(deviation)
    
    composition_score = max(0, 100 - total_deviation)
    balance_score = (completeness_score * 0.6) + (composition_score * 0.4)
    
    if balance_score >= 90:
        status = 'seimbang'
        message = '🎉 Sempurna! Piring Anda sangat seimbang!'
    elif balance_score >= 75:
        status = 'cukup_seimbang'
        message = '👍 Cukup Seimbang! Sedikit penyesuaian akan sempurna.'
    elif balance_score >= 60:
        status = 'kurang_seimbang'
        message = '⚠️ Kurang Seimbang. Perlu beberapa perbaikan.'
    else:
        status = 'tidak_seimbang'
        message = '❌ Tidak Seimbang. Tambahkan komponen yang kurang.'
    
    akg_percentage = calculate_percentage_of_akg(total_nutrition, user_type)
    
    recommendations = []
    
    if missing_classes:
        missing_str = ', '.join([NUTRITION_DB[c]['name'] for c in missing_classes])
        recommendations.append(f"Tambahkan: {missing_str}")
    
    for class_name, deviation in composition_deviations.items():
        if deviation > 10:
            recommendations.append(
                f"Kurangi {NUTRITION_DB[class_name]['name']} sekitar "
                f"{composition[class_name]['weight'] * 0.2:.0f}g"
            )
        elif deviation < -10:
            ideal_weight = (IDEAL_COMPOSITION[class_name]['percentage'] / 100) * total_weight
            need_more = ideal_weight - composition.get(class_name, {}).get('weight', 0)
            recommendations.append(
                f"Tambah {NUTRITION_DB[class_name]['name']} sekitar {need_more:.0f}g"
            )
    
    if not recommendations:
        recommendations.append("Komposisi sudah ideal! Pertahankan pola makan ini.")
    
    return {
        'total_nutrition': total_nutrition,
        'composition': composition,
        'total_weight': total_weight,
        'balance_score': balance_score,
        'status': status,
        'message': message,
        'missing_classes': list(missing_classes),
        'akg_percentage': akg_percentage,
        'recommendations': recommendations
    }

#==============================================================================
# DISPLAY FUNCTIONS
#==============================================================================

def display_nutrition_analysis(analysis, detections):
    """Display comprehensive nutrition analysis"""
    
    if analysis is None:
        st.warning("Tidak ada data nutrisi untuk ditampilkan")
        return
    
    # ⚠️ WARNING JIKA ADA KATEGORI HILANG
    if analysis['missing_classes']:
        st.markdown(f"""
        <div class="warning-box">
            <h4>⚠️ PERINGATAN: Kategori Makanan Tidak Terdeteksi</h4>
            <p>Kategori berikut <strong>TIDAK TERDETEKSI</strong> dalam gambar:</p>
            <ul>
                {''.join([f"<li>{NUTRITION_DB[c]['emoji']} <strong>{NUTRITION_DB[c]['name']}</strong></li>" for c in analysis['missing_classes']])}
            </ul>
            <p><strong>Jika Anda mengonsumsi makanan dari kategori ini, silakan foto ulang dengan menyertakan SEMUA makanan yang akan dikonsumsi dalam SATU gambar.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Balance Status
    if analysis['status'] == 'seimbang':
        st.markdown(f"""
        <div class="balance-indicator balanced">
            {analysis['message']}<br>
            Skor Keseimbangan: {analysis['balance_score']:.0f}/100
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="balance-indicator not-balanced">
            {analysis['message']}<br>
            Skor Keseimbangan: {analysis['balance_score']:.0f}/100
        </div>
        """, unsafe_allow_html=True)
    
    # Main Metrics
    st.markdown("### 📊 Ringkasan Nutrisi")
    cols = st.columns(5)
    
    metrics = [
        ('Kalori', f"{analysis['total_nutrition']['kalori']:.0f}", 'kcal'),
        ('Protein', f"{analysis['total_nutrition']['protein']:.1f}", 'g'),
        ('Lemak', f"{analysis['total_nutrition']['lemak']:.1f}", 'g'),
        ('Karbo', f"{analysis['total_nutrition']['karbohidrat']:.1f}", 'g'),
        ('Serat', f"{analysis['total_nutrition']['serat']:.1f}", 'g')
    ]
    
    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.metric(label, f"{value}{unit}")
    
    # AKG Percentage
    st.markdown("### 📈 Persentase AKG Harian")
    akg = analysis['akg_percentage']
    
    for nutrient, percentage in akg.items():
        label = nutrient.capitalize()
        st.write(f"**{label}**: {percentage:.1f}%")
        
        # Color coding for progress bar
        if percentage < 80:
            st.progress(min(percentage / 100, 1.0))
            st.caption(f"⚠️ Masih kurang {100-percentage:.1f}% dari target harian")
        elif percentage <= 120:
            st.progress(min(percentage / 100, 1.0))
            st.caption("✅ Sudah memenuhi kebutuhan harian")
        else:
            st.progress(1.0)
            st.caption(f"⚠️ Melebihi {percentage-100:.1f}% dari target harian")
    
    # Composition
    st.markdown("### 🍽️ Komposisi Piring")
    
    comp_data = []
    for class_name in CLASSES:
        if class_name in analysis['composition']:
            comp = analysis['composition'][class_name]
            ideal = IDEAL_COMPOSITION[class_name]['percentage']
            comp_data.append({
                'Kategori': NUTRITION_DB[class_name]['name'],
                'Berat (g)': f"{comp['weight']:.0f}",
                'Aktual (%)': f"{comp['percentage']:.1f}",
                'Ideal (%)': f"{ideal:.0f}",
                'Status': '✅' if abs(comp['percentage'] - ideal) < 10 else '⚠️'
            })
        else:
            comp_data.append({
                'Kategori': NUTRITION_DB[class_name]['name'],
                'Berat (g)': '0',
                'Aktual (%)': '0.0',
                'Ideal (%)': f"{IDEAL_COMPOSITION[class_name]['percentage']:.0f}",
                'Status': '❌'
            })
    
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Rekomendasi")
    for rec in analysis['recommendations']:
        st.info(f"• {rec}")
    
    # Detailed breakdown
    with st.expander("🔍 Detail Per Objek Terdeteksi"):
        for det in detections:
            st.markdown(f"""
            **{det['id']}. {NUTRITION_DB[det['class']]['emoji']} {NUTRITION_DB[det['class']]['name']}**
            - Berat Estimasi: {det['weight_grams']:.0f} gram
            - Tinggi Asumsi: {det['weight_info']['estimated_height_cm']} cm
            - Luas Area: {det['weight_info']['area_cm2']:.1f} cm²
            - Volume Estimasi: {det['weight_info']['volume_cm3']:.1f} cm³
            - Kalori: {det['nutrition']['kalori']:.0f} kcal
            - Protein: {det['nutrition']['protein']:.1f} g
            - Confidence: {det['confidence']:.2%}
            
            ---
            """)
        
        st.markdown("""
        <div class="info-box">
            <h5>ℹ️ Catatan Estimasi Berat</h5>
            <p>Estimasi berat dihitung menggunakan formula:</p>
            <p><strong>Berat = Area_2D × Tinggi_rata-rata × Densitas</strong></p>
            <p>Ini adalah <strong>PERKIRAAN AWAL</strong> dengan keterbatasan:</p>
            <ul>
                <li>Tinggi makanan diasumsikan konstan per kategori</li>
                <li>Tidak akurat untuk makanan bertumpuk tinggi</li>
                <li>Estimasi dapat berbeda 15-30% dari berat aktual</li>
            </ul>
            <p>Untuk akurasi lebih baik, gunakan timbangan digital.</p>
        </div>
        """, unsafe_allow_html=True)

#==============================================================================
# MAIN APP
#==============================================================================

def main():
    # ⭐ CRITICAL: Display warning FIRST before anything else
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ PETUNJUK PENGGUNAAN PENTING - WAJIB DIBACA!</h3>
        <h4>Untuk Hasil Analisis Nutrisi yang Akurat:</h4>
        <ol>
            <li><strong>📸 FOTO SEMUA MAKANAN dalam SATU GAMBAR</strong>
                <ul>
                    <li>Sistem hanya dapat mendeteksi objek yang terlihat di foto</li>
                    <li>Jika ada makanan yang tidak difoto, nutrisinya TIDAK akan dihitung</li>
                    <li>Pastikan SEMUA makanan dan minuman yang akan dikonsumsi terlihat jelas</li>
                </ul>
            </li>
            <li><strong>🍽️ Gunakan piring putih polos diameter 25 cm</strong> (jika memungkinkan)
                <ul>
                    <li>Untuk estimasi porsi yang lebih akurat</li>
                    <li>Jika tidak ada piring, sistem akan menggunakan fallback calibration</li>
                </ul>
            </li>
            <li><strong>📐 Ambil foto dari ATAS (top-view)</strong> dengan pencahayaan cukup
                <ul>
                    <li>Hindari bayangan yang menutupi makanan</li>
                    <li>Jarak ideal: 30-50 cm dari makanan</li>
                </ul>
            </li>
            <li><strong>✋ Pastikan makanan TIDAK saling menutupi (overlap)</strong>
                <ul>
                    <li>Susun makanan agar semua komponen terlihat jelas</li>
                    <li>Pisahkan makanan yang bertumpuk jika memungkinkan</li>
                </ul>
            </li>
        </ol>
        <h4>⚠️ DISCLAIMER - Keterbatasan Sistem:</h4>
        <ul>
            <li><strong>Estimasi berat bersifat PERKIRAAN</strong> berdasarkan segmentasi 2D</li>
            <li>Formula: <code>Berat = Area_2D × Tinggi_asumsi × Densitas_asumsi</code></li>
            <li>Tinggi dan densitas diasumsikan konstan per kategori makanan</li>
            <li><strong>Akurasi estimasi: ±15-30%</strong> dari berat aktual</li>
            <li>Tidak akurat untuk makanan bertumpuk tinggi atau bentuk 3D kompleks</li>
            <li>Sistem ini adalah <strong>baseline research</strong> untuk pengembangan selanjutnya</li>
        </ul>
        <p><em>Untuk kebutuhan medis atau diet ketat, konsultasikan dengan ahli gizi profesional.</em></p>
    </div>
    """, unsafe_allow_html=True)
    
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ SmartPlate")
        st.markdown("**Nutrition Balance Detector**")
        st.markdown("Powered by YOLOv8 Segmentation")
        
        st.divider()
        
        st.markdown("### ⚙️ Pengaturan")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        
        user_type = st.selectbox(
            "Profil Pengguna",
            ['male_adult', 'female_adult', 'child', 'pregnant_trimester1', 'pregnant_trimester2', 'pregnant_trimester3'],
            format_func=lambda x: {
                'male_adult': '👨 Dewasa Laki-laki',
                'female_adult': '👩 Dewasa Perempuan',
                'child': '🧒 Anak (10-12 tahun)',
                'pregnant_trimester1': '🤰 Ibu Hamil Trimester 1',
                'pregnant_trimester2': '🤰 Ibu Hamil Trimester 2',
                'pregnant_trimester3': '🤰 Ibu Hamil Trimester 3'
            }[x]
        )
        
        # Show AKG for selected profile
        st.markdown("### 📊 AKG Target Harian")
        akg_target = AKG_DATABASE[user_type]
        st.write(f"Kalori: {akg_target['kalori']} kkal")
        st.write(f"Protein: {akg_target['protein']} g")
        st.write(f"Karbohidrat: {akg_target['karbohidrat']} g")
        st.write(f"Lemak: {akg_target['lemak']} g")
        st.write(f"Serat: {akg_target['serat']} g")
        
        st.divider()
        
        st.markdown("### 📚 Kategori Deteksi")
        for class_name in CLASSES:
            info = NUTRITION_DB[class_name]
            st.markdown(f"**{info['emoji']} {info['name']}**")
        
        st.divider()
        
        st.markdown("### ℹ️ Tentang Sistem")
        st.markdown("""
        **Versi:** 2.0  
        **Model:** YOLOv8-seg  
        **Dataset:** FoodSeg103 + Indonesia  
        **Kategori:** 4 Sehat 5 Sempurna  
        
        **Improvisasi Model:**
        - Copy-paste augmentation
        - Mixup augmentation
        - Random erasing
        - Multi-scale training
        - Cosine LR scheduling
        """)
    
    # Main App
    st.title("🍽️ SmartPlate - Nutrition Balance Detector")
    st.markdown("### Analisis Nutrisi dengan Instance Segmentation")
    
    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📸 Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload gambar piring makanan", 
                                          type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Gambar Original", use_container_width=True)
            
            if st.button("🔍 Analisis Nutrisi", type="primary"):
                with st.spinner("⏳ Sedang menganalisis makanan Anda..."):
                    results = model.predict(
                        source=image, 
                        conf=conf_threshold, 
                        iou=IOU_THRESHOLD, 
                        verbose=False
                    )
                    
                    annotated_img, detections, pixel_to_cm, plate_detected = process_segmentation_results(
                        image, results, conf_threshold
                    )
                    
                    st.session_state['annotated_img'] = annotated_img
                    st.session_state['detections'] = detections
                    st.session_state['pixel_to_cm'] = pixel_to_cm
                    st.session_state['plate_detected'] = plate_detected
                    st.session_state['analysis'] = analyze_nutrition_balance(
                        detections, user_type
                    )
            
            if 'annotated_img' in st.session_state:
                with col2:
                    st.image(st.session_state['annotated_img'],
                           caption="Hasil Segmentasi & Deteksi", use_container_width=True)
                
                # Info tentang kalibrasi
                if st.session_state['plate_detected']:
                    st.success(f"✅ Piring terdeteksi! Kalibrasi: {st.session_state['pixel_to_cm']:.4f} cm/pixel")
                else:
                    st.warning(f"⚠️ Piring tidak terdeteksi. Menggunakan fallback calibration: {st.session_state['pixel_to_cm']:.4f} cm/pixel")
                    st.info("💡 Untuk estimasi lebih akurat, gunakan piring putih polos diameter 25 cm")
                
                st.divider()
                
                if st.session_state['detections']:
                    display_nutrition_analysis(
                        st.session_state['analysis'],
                        st.session_state['detections']
                    )
                else:
                    st.markdown("""
                    <div class="error-box">
                        <h4>❌ Tidak Ada Makanan Terdeteksi</h4>
                        <p>Sistem tidak dapat mendeteksi makanan dalam gambar. Kemungkinan penyebab:</p>
                        <ul>
                            <li>Pencahayaan terlalu gelap atau terlalu terang</li>
                            <li>Makanan terlalu kecil atau terlalu jauh</li>
                            <li>Makanan tidak termasuk dalam 5 kategori yang didukung</li>
                            <li>Foto terlalu blur atau tidak fokus</li>
                        </ul>
                        <p><strong>Saran:</strong> Coba foto ulang dengan pencahayaan lebih baik dan jarak lebih dekat (30-50 cm).</p>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        camera_img = st.camera_input("Ambil foto makanan Anda")
        
        if camera_img:
            image = Image.open(camera_img)
            
            with st.spinner("⏳ Menganalisis foto dari kamera..."):
                results = model.predict(
                    source=image, 
                    conf=conf_threshold,
                    iou=IOU_THRESHOLD, 
                    verbose=False
                )
                
                annotated_img, detections, pixel_to_cm, plate_detected = process_segmentation_results(
                    image, results, conf_threshold
                )
                
                st.image(annotated_img, caption="Hasil Analisis", use_container_width=True)
                
                # Info kalibrasi
                if plate_detected:
                    st.success(f"✅ Piring terdeteksi! Kalibrasi: {pixel_to_cm:.4f} cm/pixel")
                else:
                    st.warning(f"⚠️ Piring tidak terdeteksi. Fallback calibration: {pixel_to_cm:.4f} cm/pixel")
                
                st.divider()
                
                if detections:
                    analysis = analyze_nutrition_balance(detections, user_type)
                    display_nutrition_analysis(analysis, detections)
                else:
                    st.error("❌ Tidak ada makanan terdeteksi. Silakan foto ulang dengan pencahayaan lebih baik.")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p><strong>SmartPlate v2.0</strong> - Nutrition Balance Detector</p>
        <p>© 2025 Mochamad Faisal Akbar | Powered by YOLOv8-seg</p>
        <p><em>Sistem ini menggunakan estimasi berbasis computer vision. Untuk kebutuhan medis, konsultasikan dengan ahli gizi.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()