"""
THE GOLDEN VECTOR ENGINE: Phase 1
Objective: Extract and average pose-invariant ratios from the 20-image Ground Truth dataset.
Output: master_keys.json (The Master Key payload for Android)
"""
import os
import cv2
import json
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
DATA_DIR = "dataset"  # Ensure your SQUARE, ROUND, OVAL, OBLONG folders are inside here
OUTPUT_JSON = "master_keys.json"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "../assets")
MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print("[SYSTEM] Fetching MediaPipe Task Model...")
        os.makedirs(MODEL_DIR, exist_ok=True)
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)

ensure_model_exists()

# Initialize Task API
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    output_face_blendshapes=False, 
    output_facial_transformation_matrixes=False, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

def calc_dist(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_ratios(image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = detector.detect(mp_image)
    
    if not results.face_landmarks: return None
    lm = results.face_landmarks[0]
    
    cheek_w = calc_dist(lm[234], lm[454])
    if cheek_w == 0: return None
    
    face_l = calc_dist(lm[10], lm[152])
    forehead_w = calc_dist(lm[103], lm[332])
    upper_jaw_w = calc_dist(lm[132], lm[361])
    lower_jaw_w = calc_dist(lm[136], lm[365])
    
    return [
        face_l / cheek_w,             # Aspect Ratio
        forehead_w / cheek_w,         # Forehead Ratio
        upper_jaw_w / cheek_w,        # Upper Jaw Ratio
        lower_jaw_w / cheek_w,        # Lower Jaw Ratio
        (upper_jaw_w - lower_jaw_w) / cheek_w  # Taper Delta
    ]

def forge_keys():
    print("[+] INITIATING MASTER KEY FORGE...")
    target_shapes = ["SQUARE", "ROUND", "OVAL", "OBLONG"]
    master_keys = {}

    for shape in target_shapes:
        shape_path = os.path.join(DATA_DIR, shape)
        if not os.path.exists(shape_path):
            print(f"[-] WARNING: Directory {shape} not found.")
            continue
            
        print(f"[*] Extracting telemetry for {shape}...")
        shape_vectors = []
        
        for img_name in os.listdir(shape_path):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')): 
                continue
                
            img_path = os.path.join(shape_path, img_name)
            vector = extract_ratios(img_path)
            
            if vector:
                shape_vectors.append(vector)
            else:
                print(f"    [-] FAILED to map: {img_name}")
                
        if shape_vectors:
            # Average the vectors to create the Golden Vector
            golden_vector = np.mean(shape_vectors, axis=0).tolist()
            # Rounding to 4 decimal places for clean JSON formatting
            master_keys[shape] = [round(num, 4) for num in golden_vector]
            print(f"    [+] {shape} Golden Vector Forged: {len(shape_vectors)} assets merged.")

    # Secure the payload
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(master_keys, f, indent=4)
        
    print(f"\n[+] VAULT LOCKED. Master Keys saved to: {OUTPUT_JSON}")

if __name__ == "__main__":
    forge_keys()