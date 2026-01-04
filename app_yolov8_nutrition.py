import os
import streamlit as st
import numpy as np
from PIL import Image
import cv2
import gdown

# Page config
st.set_page_config(
    page_title="SmartPlate - Nutrition Segmentation",
    page_icon="🍽️",
    layout="wide"
)

# Model configuration
MODEL_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID'  # ⚠️ CHANGE THIS
MODEL_PATH = 'best_nutrition_segmentation.pt'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Classes
CLASSES = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
CLASS_COLORS = {
    0: (255, 107, 107),  # buah - red
    1: (249, 202, 36),   # karbohidrat - yellow
    2: (52, 152, 219),   # minuman - blue
    3: (225, 112, 85),   # protein - orange
    4: (0, 184, 148)     # sayur - green
}

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {width: 100%; background-color: #0245d6; color: white;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load YOLOv8 segmentation model"""
    try:
        from ultralytics import YOLO
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('📥 Downloading model...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success('✅ Model downloaded!')
        
        with st.spinner('🔄 Loading model...'):
            model = YOLO(MODEL_PATH)
            st.success('✅ Model loaded!')
        
        return model
    
    except ImportError:
        st.error("❌ Run: pip install ultralytics")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def process_results(image, results, conf_threshold):
    """Process and visualize results - SIMPLIFIED VERSION"""
    img_array = np.array(image).copy()
    overlay = img_array.copy()
    detections = []
    
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        return Image.fromarray(img_array), []
    
    # Process each detection
    for i, (box, mask, cls, conf) in enumerate(zip(
        result.boxes.xyxy, result.masks.data, result.boxes.cls, result.boxes.conf
    )):
        if conf < conf_threshold:
            continue
        
        class_idx = int(cls)
        class_name = CLASSES[class_idx]
        confidence = float(conf)
        
        # Get mask
        mask_array = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
        
        # Color
        import matplotlib.pyplot as plt
        color = plt.cm.tab10(class_idx)[:3]
        color_rgb = tuple([int(c * 255) for c in color])
        
        # Draw mask overlay
        mask_bool = mask_resized > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * 0.6 + color_rgb[c] * 0.4,
                overlay[:, :, c]
            )
        
        # Draw box
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color_rgb, 2)
        
        # Label
        label = f"{class_name}: {confidence:.2%}"
        cv2.rectangle(img_array, (x1, y1-25), (x1+150, y1), color_rgb, -1)
        cv2.putText(img_array, label, (x1+5, y1-8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        
        detections.append({
            'class': class_name,
            'confidence': confidence
        })
    
    # Blend
    final_img = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    return Image.fromarray(final_img), detections

def main():
    model = load_model()
    
    if model is None:
        st.stop()
    
    st.title("🍽️ SmartPlate - Nutrition Segmentation")
    st.markdown("### Deteksi Makanan dengan YOLOv8 Instance Segmentation")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.25, 0.05)
        
        st.divider()
        st.markdown("### 📚 Categories")
        for i, cls in enumerate(CLASSES):
            color_hex = '#' + ''.join([f'{c:02x}' for c in CLASS_COLORS[i]])
            st.markdown(f"<span style='color:{color_hex}'>●</span> {cls}", 
                       unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2 = st.tabs(["📤 Upload", "📸 Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            col1, col2 = st.columns(2)
            
            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Original", use_container_width=True)
            
            if st.button("🔍 Detect", type="primary"):
                with st.spinner("Analyzing..."):
                    results = model.predict(
                        source=image, 
                        conf=conf_threshold,
                        iou=IOU_THRESHOLD,
                        verbose=False
                    )
                    
                    annotated_img, detections = process_results(
                        image, results, conf_threshold
                    )
                    
                    with col2:
                        st.image(annotated_img, caption="Result", 
                               use_container_width=True)
                    
                    if detections:
                        st.success(f"✅ Detected {len(detections)} objects")
                        for i, det in enumerate(detections, 1):
                            st.write(f"{i}. **{det['class']}** - {det['confidence']:.1%}")
                    else:
                        st.warning("No objects detected")
    
    with tab2:
        camera_img = st.camera_input("Take photo")
        
        if camera_img:
            image = Image.open(camera_img)
            
            with st.spinner("Analyzing..."):
                results = model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=IOU_THRESHOLD,
                    verbose=False
                )
                
                annotated_img, detections = process_results(
                    image, results, conf_threshold
                )
                
                st.image(annotated_img, use_container_width=True)
                
                if detections:
                    st.success(f"✅ Detected {len(detections)} objects")
                    for i, det in enumerate(detections, 1):
                        st.write(f"{i}. **{det['class']}** - {det['confidence']:.1%}")
    
    st.divider()
    st.markdown("© 2024 SmartPlate | Powered by YOLOv8-seg")

if __name__ == '__main__':
    main()

# Page config
st.set_page_config(
    page_title="SmartPlate - Nutrition Segmentation",
    page_icon="🍽️",
    layout="wide"
)

# Model configuration
MODEL_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID'  # ⚠️ CHANGE THIS
MODEL_PATH = 'best_nutrition_segmentation.pt'
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Classes
CLASSES = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 20px;}
    .stButton>button {width: 100%; background-color: #0245d6; color: white;}
    .nutrition-card {padding: 15px; border-radius: 10px; margin: 10px 0; 
                     box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
    .balance-indicator {padding: 20px; border-radius: 10px; text-align: center;
                        font-size: 24px; font-weight: bold; margin: 20px 0;}
    .balanced {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;}
    .not-balanced {background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load YOLOv8 segmentation model"""
    try:
        from ultralytics import YOLO
        
        if not os.path.exists(MODEL_PATH):
            with st.spinner('📥 Downloading model...'):
                url = f'https://drive.google.com/uc?id={MODEL_ID}'
                gdown.download(url, MODEL_PATH, quiet=False)
                st.success('✅ Model downloaded!')
        
        with st.spinner('🔄 Loading YOLOv8 segmentation model...'):
            model = YOLO(MODEL_PATH)
            st.success('✅ Model loaded!')
        
        return model
    
    except ImportError:
        st.error("❌ Ultralytics not installed. Run: pip install ultralytics")
        return None
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def detect_plate_and_calibrate(image):
    """
    Detect circular plate for calibration
    Returns pixel_to_cm ratio
    """
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
    
    # Default fallback
    return 0.05, None  # Assume 1 pixel ≈ 0.05 cm

def calculate_weight_from_mask(mask_array, pixel_to_cm, class_name):
    """
    Convert segmentation mask to weight (grams)
    
    Steps:
    1. Count mask pixels
    2. Convert to cm² using calibration
    3. Estimate volume (area × avg_height)
    4. Convert to grams (volume × density)
    """
    # Count pixels in mask
    mask_pixels = np.sum(mask_array > 0)
    
    # Convert to area (cm²)
    area_cm2 = mask_pixels * (pixel_to_cm ** 2)
    
    # Get average height for this class (assumption)
    avg_heights = {
        'karbohidrat': 2.5,  # cm
        'protein': 3.0,
        'sayur': 2.0,
        'buah': 3.5,
        'minuman': 10.0  # height in glass
    }
    height_cm = avg_heights.get(class_name, 2.5)
    
    # Calculate volume
    volume_cm3 = area_cm2 * height_cm
    
    # Get density and calculate weight
    density = NUTRITION_DB[class_name]['density']
    weight_grams = volume_cm3 * density
    
    return {
        'mask_pixels': int(mask_pixels),
        'area_cm2': float(area_cm2),
        'volume_cm3': float(volume_cm3),
        'weight_grams': float(weight_grams)
    }

def process_segmentation_results(image, results, conf_threshold=0.25):
    """
    Process YOLOv8 segmentation results
    Returns: annotated image, detection list, nutrition analysis
    """
    # Calibrate using plate detection
    pixel_to_cm, plate_circle = detect_plate_and_calibrate(image)
    
    img_array = np.array(image).copy()
    overlay = img_array.copy()
    detections = []
    
    result = results[0]
    
    if result.masks is None or len(result.masks) == 0:
        return Image.fromarray(img_array), [], None
    
    # Process each detection
    for i, (box, mask, cls, conf) in enumerate(zip(
        result.boxes.xyxy, result.masks.data, result.boxes.cls, result.boxes.conf
    )):
        if conf < conf_threshold:
            continue
        
        class_idx = int(cls)
        class_name = CLASSES[class_idx]
        confidence = float(conf)
        
        # Get mask and resize to image size
        mask_array = mask.cpu().numpy()
        mask_resized = cv2.resize(mask_array, (img_array.shape[1], img_array.shape[0]))
        
        # Calculate weight from mask
        weight_info = calculate_weight_from_mask(mask_resized, pixel_to_cm, class_name)
        weight_grams = weight_info['weight_grams']
        
        # Calculate nutrition
        nutrition = calculate_nutrition_from_grams(class_name, weight_grams)
        
        # Get color for this class
        color = plt.cm.tab10(class_idx)[:3]
        color_rgb = tuple([int(c * 255) for c in color])
        
        # Draw segmentation mask (overlay)
        mask_bool = mask_resized > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                overlay[:, :, c] * 0.6 + color_rgb[c] * 0.4,
                overlay[:, :, c]
            )
        
        # Draw bounding box
        x1, y1, x2, y2 = box.cpu().numpy().astype(int)
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color_rgb, 2)
        
        # Draw label with nutrition info
        label = f"{class_name}: {weight_grams:.0f}g, {nutrition['kalori']:.0f}kcal"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        cv2.rectangle(img_array, (x1, y1 - text_h - 10), 
                     (x1 + text_w, y1), color_rgb, -1)
        cv2.putText(img_array, label, (x1, y1 - 5),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Store detection info
        detections.append({
            'id': i + 1,
            'class': class_name,
            'confidence': confidence,
            'bbox': [x1, y1, x2, y2],
            'mask_pixels': weight_info['mask_pixels'],
            'area_cm2': weight_info['area_cm2'],
            'volume_cm3': weight_info['volume_cm3'],
            'weight_grams': weight_grams,
            'nutrition': nutrition
        })
    
    # Blend original with overlay
    final_img = cv2.addWeighted(img_array, 0.7, overlay, 0.3, 0)
    
    # Draw plate circle if detected
    if plate_circle is not None:
        cx, cy, r = plate_circle
        cv2.circle(final_img, (cx, cy), r, (0, 255, 0), 2)
        cv2.putText(final_img, f"Plate: {r*2*pixel_to_cm:.1f}cm", 
                   (cx-50, cy-r-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (0, 255, 0), 2)
    
    return Image.fromarray(final_img), detections, pixel_to_cm

def analyze_nutrition_balance(detections, user_type='male_adult'):
    """Comprehensive nutrition analysis"""
    
    if not detections:
        return None
    
    # Aggregate nutrition
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
        
        # Sum nutrition
        for key in total_nutrition:
            if key in nutrition:
                total_nutrition[key] += nutrition[key]
        
        # Track composition
        if class_name not in composition:
            composition[class_name] = {'weight': 0, 'count': 0}
        composition[class_name]['weight'] += weight
        composition[class_name]['count'] += 1
        total_weight += weight
    
    # Calculate percentages
    for class_name in composition:
        composition[class_name]['percentage'] = \
            (composition[class_name]['weight'] / total_weight * 100) if total_weight > 0 else 0
    
    # Check completeness
    detected_classes = set(composition.keys())
    required_classes = set(CLASSES)
    missing_classes = required_classes - detected_classes
    completeness_score = (len(detected_classes) / len(required_classes)) * 100
    
    # Check composition deviation from ideal
    composition_deviations = {}
    total_deviation = 0
    
    for class_name in CLASSES:
        ideal_pct = IDEAL_COMPOSITION[class_name]['percentage']
        actual_pct = composition.get(class_name, {}).get('percentage', 0)
        deviation = actual_pct - ideal_pct
        composition_deviations[class_name] = deviation
        total_deviation += abs(deviation)
    
    composition_score = max(0, 100 - total_deviation)
    
    # Final balance score
    balance_score = (completeness_score * 0.6) + (composition_score * 0.4)
    
    # Determine status
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
    
    # AKG percentage
    akg_percentage = calculate_percentage_of_akg(total_nutrition, user_type)
    
    # Generate recommendations
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
        'completeness_score': completeness_score,
        'composition_score': composition_score,
        'balance_score': balance_score,
        'status': status,
        'message': message,
        'missing_classes': list(missing_classes),
        'deviations': composition_deviations,
        'akg_percentage': akg_percentage,
        'recommendations': recommendations
    }

def display_nutrition_analysis(analysis, detections):
    """Display comprehensive nutrition analysis"""
    
    if analysis is None:
        st.warning("Tidak ada data nutrisi untuk ditampilkan")
        return
    
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
        st.progress(min(percentage / 100, 1.0))
    
    st.caption("*AKG: Angka Kecukupan Gizi (untuk dewasa laki-laki)")
    
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
    with st.expander("🔍 Detail Per Objek"):
        for det in detections:
            st.markdown(f"""
            **{det['id']}. {NUTRITION_DB[det['class']]['emoji']} {NUTRITION_DB[det['class']]['name']}**
            - Berat: {det['weight_grams']:.0f} gram
            - Kalori: {det['nutrition']['kalori']:.0f} kcal
            - Protein: {det['nutrition']['protein']:.1f} g
            - Karbohidrat: {det['nutrition']['karbohidrat']:.1f} g
            - Confidence: {det['confidence']:.2%}
            """)

def main():
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.title("ℹ️ SmartPlate")
        st.markdown("**Nutrition Balance Detector**")
        st.markdown("Menggunakan YOLOv8 Instance Segmentation")
        
        st.divider()
        
        st.markdown("### ⚙️ Pengaturan")
        conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
        
        user_type = st.selectbox(
            "Tipe Pengguna (untuk AKG)",
            ['male_adult', 'female_adult', 'child'],
            format_func=lambda x: {
                'male_adult': 'Dewasa Laki-laki',
                'female_adult': 'Dewasa Perempuan',
                'child': 'Anak (10-12 tahun)'
            }[x]
        )
        
        st.divider()
        
        st.markdown("### 📚 Kategori")
        for class_name in CLASSES:
            info = NUTRITION_DB[class_name]
            st.markdown(f"**{info['emoji']} {info['name']}**")
    
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
            
            if st.button("🔍 Analisis", type="primary"):
                with st.spinner("Sedang menganalisis..."):
                    results = model.predict(source=image, conf=conf_threshold, 
                                           iou=IOU_THRESHOLD, verbose=False)
                    
                    annotated_img, detections, pixel_to_cm = \
                        process_segmentation_results(image, results, conf_threshold)
                    
                    st.session_state['annotated_img'] = annotated_img
                    st.session_state['detections'] = detections
                    st.session_state['analysis'] = analyze_nutrition_balance(
                        detections, user_type
                    )
            
            if 'annotated_img' in st.session_state:
                with col2:
                    st.image(st.session_state['annotated_img'],
                           caption="Hasil Segmentation", use_container_width=True)
                
                st.divider()
                display_nutrition_analysis(
                    st.session_state['analysis'],
                    st.session_state['detections']
                )
    
    with tab2:
        camera_img = st.camera_input("Ambil foto")
        
        if camera_img:
            image = Image.open(camera_img)
            
            with st.spinner("Menganalisis..."):
                results = model.predict(source=image, conf=conf_threshold,
                                       iou=IOU_THRESHOLD, verbose=False)
                
                annotated_img, detections, _ = process_segmentation_results(
                    image, results, conf_threshold
                )
                
                st.image(annotated_img, use_container_width=True)
                
                analysis = analyze_nutrition_balance(detections, user_type)
                display_nutrition_analysis(analysis, detections)
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center;">
        <p>© 2024 Andromeda Team | Powered by YOLOv8-seg</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()