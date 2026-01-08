#==============================================================================
# IMPROVED YOLOV8 INSTANCE SEGMENTATION FOR NUTRITION ANALYSIS
# Version 2.0 - Performance Optimizations
#==============================================================================

import os
import yaml
import shutil
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
from IPython.display import FileLink

# Install Ultralytics
!pip install ultralytics -q

from ultralytics import YOLO
import torch

#==============================================================================
# CONFIGURATION
#==============================================================================

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")

BASE_DIR = '/kaggle/input/nutrition-dataset-yolo'
WORK_DIR = '/kaggle/working'
OUTPUT_DIR = os.path.join(WORK_DIR, 'yolo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASSES = ['buah', 'karbohidrat', 'minuman', 'protein', 'sayur']
NUM_CLASSES = len(CLASSES)

# ⭐ CLASS WEIGHTS (Handle imbalance)
# Based on training output:
# buah: 4241, karbohidrat: 5950, minuman: 1437, protein: 5345, sayur: 9962
CLASS_WEIGHTS = {
    0: 1.0,   # buah
    1: 1.0,   # karbohidrat
    2: 3.5,   # minuman (boost - lowest count)
    3: 1.2,   # protein
    4: 0.8    # sayur (reduce - highest count)
}

#==============================================================================
# VERIFY DATASET
#==============================================================================

def verify_yolo_segmentation_format(data_dir):
    print("\n" + "="*80)
    print("VERIFYING DATASET FORMAT")
    print("="*80)
    
    required_dirs = [
        'images/train', 'images/val', 'images/test',
        'labels/train', 'labels/val', 'labels/test'
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if os.path.exists(full_path):
            count = len([f for f in os.listdir(full_path) 
                        if not f.startswith('.')])
            print(f"✅ {dir_path:20s}: {count} files")
        else:
            print(f"❌ {dir_path:20s}: NOT FOUND")
            return False
    
    yaml_path = os.path.join(data_dir, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"✅ data.yaml found")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            print(f"   Classes: {data.get('names', 'Not found')}")
            print(f"   NC: {data.get('nc', 'Not found')}")
    else:
        print(f"❌ data.yaml NOT FOUND")
        return False
    
    return True

data_yaml_path = os.path.join(BASE_DIR, 'data.yaml')
if not verify_yolo_segmentation_format(BASE_DIR):
    print("\n❌ Dataset format has issues!")
    raise Exception("Fix dataset format before training")

print("\n✅ Dataset format is correct!")

#==============================================================================
# TRAIN IMPROVED MODEL
#==============================================================================

print("\n" + "="*80)
print("TRAINING YOLOV8 SEGMENTATION MODEL - IMPROVED VERSION")
print("="*80)

model = YOLO('yolov8s-seg.pt')

print(f"\n📦 Model: YOLOv8s-seg")
print(f"   Parameters: ~11M")
print(f"   Size: ~22 MB")

# ⭐ IMPROVED CONFIGURATION
EPOCHS = 150          # ← Increased from 100
IMG_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 30         # ← Increased from 20

print(f"\n⚙️ Training Config:")
print(f"   Epochs: {EPOCHS}")
print(f"   Image Size: {IMG_SIZE}")
print(f"   Batch Size: {BATCH_SIZE}")
print(f"   Early Stopping: {PATIENCE} epochs")
print(f"   Class Weights: {CLASS_WEIGHTS}")

# TRAIN
results = model.train(
    data=data_yaml_path,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    device=0,
    workers=4,
    project=OUTPUT_DIR,
    name='nutrition_segmentation_v2',
    exist_ok=True,
    
    # ⭐ IMPROVED OPTIMIZER
    optimizer='AdamW',     # Better than Adam
    lr0=0.001,
    lrf=0.001,             # Lower final LR
    momentum=0.937,
    weight_decay=0.0005,
    
    # Loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,
    
    # ⭐ IMPROVED AUGMENTATION
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,          # Increased rotation
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,             # ⭐ NEW: Mix images
    copy_paste=0.2,        # ⭐ NEW: Copy objects
    erasing=0.4,           # ⭐ NEW: Random erasing
    
    # ⭐ MULTI-SCALE TRAINING
    multi_scale=True,      # Train with varying image sizes
    
    # ⭐ COSINE LR SCHEDULE
    cos_lr=True,           # Cosine learning rate
    
    # Other
    save=True,
    save_period=10,
    cache=False,
    plots=True,
    verbose=True
)

print("\n✅ Training complete!")

#==============================================================================
# EVALUATE MODEL
#==============================================================================

print("\n" + "="*80)
print("EVALUATING MODEL")
print("="*80)

best_model_path = os.path.join(OUTPUT_DIR, 'nutrition_segmentation_v2', 'weights', 'best.pt')
best_model = YOLO(best_model_path)

test_results = best_model.val(
    data=data_yaml_path,
    split='test',
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    plots=True,
    save_json=True,
    project=OUTPUT_DIR,
    name='test_evaluation_v2'
)

print(f"\n📊 Test Results:")
print(f"   Box mAP50: {test_results.box.map50:.4f}")
print(f"   Box mAP50-95: {test_results.box.map:.4f}")
print(f"   Mask mAP50: {test_results.seg.map50:.4f}")
print(f"   Mask mAP50-95: {test_results.seg.map:.4f}")
print(f"   Precision: {test_results.box.mp:.4f}")
print(f"   Recall: {test_results.box.mr:.4f}")

print(f"\n📈 Per-Class Segmentation mAP50:")
for i, class_name in enumerate(CLASSES):
    print(f"   {class_name:15s}: {test_results.seg.maps[i]:.4f}")

#==============================================================================
# INFERENCE SAMPLES
#==============================================================================

print("\n" + "="*80)
print("RUNNING INFERENCE")
print("="*80)

test_images_dir = os.path.join(BASE_DIR, 'images', 'test')
test_images = [os.path.join(test_images_dir, f) 
               for f in os.listdir(test_images_dir)[:5]]

inference_results = best_model.predict(
    source=test_images,
    imgsz=IMG_SIZE,
    conf=0.25,
    iou=0.45,
    save=True,
    project=OUTPUT_DIR,
    name='inference_samples_v2',
    show_labels=True,
    show_conf=True,
    show_boxes=True
)

print(f"✅ Inference complete!")

#==============================================================================
# EXPORT MODEL
#==============================================================================

print("\n" + "="*80)
print("EXPORTING MODEL")
print("="*80)

# Export to ONNX
try:
    onnx_path = best_model.export(format='onnx', dynamic=True)
    print(f"✅ Exported to ONNX: {onnx_path}")
except Exception as e:
    print(f"⚠️ ONNX export failed: {e}")

# Copy to working directory
shutil.copy2(best_model_path, os.path.join(WORK_DIR, 'best_nutrition_segmentation_v2.pt'))
print(f"✅ Copied to: {WORK_DIR}/best_nutrition_segmentation_v2.pt")

#==============================================================================
# CREATE SUMMARY
#==============================================================================

print("\n" + "="*80)
print("CREATING SUMMARY REPORT")
print("="*80)

summary = {
    'Model': 'YOLOv8s-seg (Improved)',
    'Version': '2.0',
    'Epochs Trained': EPOCHS,
    'Image Size': IMG_SIZE,
    'Batch Size': BATCH_SIZE,
    'Optimizer': 'AdamW',
    'Multi-Scale': 'Yes',
    'Cosine LR': 'Yes',
    'Mixup': 0.1,
    'Copy-Paste': 0.2,
    'Box mAP50': f"{test_results.box.map50:.4f}",
    'Box mAP50-95': f"{test_results.box.map:.4f}",
    'Mask mAP50': f"{test_results.seg.map50:.4f}",
    'Mask mAP50-95': f"{test_results.seg.map:.4f}",
    'Precision': f"{test_results.box.mp:.4f}",
    'Recall': f"{test_results.box.mr:.4f}",
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(os.path.join(WORK_DIR, 'segmentation_summary_v2.csv'), index=False)

print("\n📄 Training Summary:")
for key, value in summary.items():
    print(f"   {key:20s}: {value}")

# Per-class metrics
class_metrics = pd.DataFrame({
    'Class': CLASSES,
    'Box_mAP50': test_results.box.maps,
    'Mask_mAP50': test_results.seg.maps
})
class_metrics.to_csv(os.path.join(WORK_DIR, 'class_metrics_v2.csv'), index=False)

#==============================================================================
# ⭐ CREATE DOWNLOAD LINKS (KAGGLE SPECIFIC)
#==============================================================================

print("\n" + "="*80)
print("CREATING DOWNLOAD LINKS")
print("="*80)

# Create download directory
download_dir = os.path.join(WORK_DIR, 'download_package')
os.makedirs(download_dir, exist_ok=True)

# Copy important files
files_to_download = [
    ('best_nutrition_segmentation_v2.pt', 'Model weights'),
    ('segmentation_summary_v2.csv', 'Training summary'),
    ('class_metrics_v2.csv', 'Per-class metrics')
]

for filename, description in files_to_download:
    src = os.path.join(WORK_DIR, filename)
    dst = os.path.join(download_dir, filename)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"✅ {description}: {filename}")

# Display download links
print("\n📥 Download Files:")
for filename, _ in files_to_download:
    filepath = os.path.join(download_dir, filename)
    if os.path.exists(filepath):
        display(FileLink(filepath))

print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE! 🎉")
print("="*80)
print(f"\n🎯 Performance Targets:")
print(f"   Mask mAP50 > 0.80: {'✅ ACHIEVED' if test_results.seg.map50 > 0.80 else '⚠️ CURRENT: ' + f'{test_results.seg.map50:.4f}'}")
print(f"   Mask mAP50-95 > 0.60: {'✅ ACHIEVED' if test_results.seg.map > 0.60 else '⚠️ CURRENT: ' + f'{test_results.seg.map:.4f}'}")

print(f"\n💡 Next Steps:")
print(f"   1. Download best_nutrition_segmentation_v2.pt from download_package/")
print(f"   2. Upload to Google Drive")
print(f"   3. Get shareable link & extract File ID")
print(f"   4. Update MODEL_ID in Streamlit app")
print(f"   5. Deploy to Streamlit Cloud")

print(f"\n📊 Improvement Suggestions:")
if test_results.seg.map < 0.60:
    print(f"   • Consider training for 200+ epochs")
    print(f"   • Try YOLOv8m-seg (larger model)")
    print(f"   • Add more training data for protein & minuman classes")
    print(f"   • Increase mixup to 0.2 for better generalization")