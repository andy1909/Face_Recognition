# -----------------------------------------------------------------------------
# FILE: app.py
# MÔ TẢ: Flask server cho ứng dụng nhận diện khuôn mặt.
# PHIÊN BẢN: Đã sửa lỗi tương thích kiểu dữ liệu giữa MTCNN và OpenCV.
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from keras.models import load_model
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from mtcnn.mtcnn import MTCNN

# --- 1. CẤU HÌNH ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
CLASS_NAMES = ['hoan', 'long', 'nghia']
MODEL_PATH = 'face_model.h5'
UPLOAD_FOLDER = 'uploads'

# --- 2. KHỞI TẠO ỨNG DỤNG FLASK VÀ CÁC MÔ HÌNH ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    recognition_model = load_model(MODEL_PATH)
    print(f"[INFO] Đã tải thành công mô hình nhận diện từ '{MODEL_PATH}'")
except Exception as e:
    recognition_model = None
    print(f"[ERROR] Không thể tải mô hình nhận diện: {e}")

try:
    face_detector = MTCNN()
    print("[INFO] Đã tải thành công mô hình phát hiện khuôn mặt MTCNN.")
except Exception as e:
    face_detector = None
    print(f"[ERROR] Không thể khởi tạo MTCNN: {e}")


# --- 3. LOGIC XỬ LÝ ẢNH ---
def align_face(img, left_eye, right_eye):
    """Xoay ảnh để căn chỉnh khuôn mặt dựa trên vị trí hai mắt."""
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # SỬA LỖI: Ép kiểu tọa độ tâm về int chuẩn của Python
    center_x = int((left_eye[0] + right_eye[0]) / 2)
    center_y = int((left_eye[1] + right_eye[1]) / 2)
    center = (center_x, center_y)
    
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    (w, h) = img.shape[1], img.shape[0]
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return aligned_img

def detect_and_recognize_faces(image_path):
    """Phát hiện, căn chỉnh và nhận diện các khuôn mặt trong ảnh."""
    if face_detector is None or recognition_model is None:
        raise RuntimeError("Các mô hình chưa được khởi tạo đúng cách.")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return [], (0, 0)
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    original_height, original_width = img_rgb.shape[:2]
    
    detections = face_detector.detect_faces(img_rgb)
    
    results = []
    for det in detections:
        if det['confidence'] < 0.95: continue
            
        x, y, w, h = det['box']
        keypoints = det['keypoints']
        
        aligned_face = align_face(img_bgr, keypoints['left_eye'], keypoints['right_eye'])
        
        face_crop = aligned_face[y:y+h, x:x+w]
        
        if face_crop.size == 0: continue # Bỏ qua nếu cắt bị lỗi

        # Chuyển ảnh crop sang ảnh xám để tương thích với model
        face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        # Resize ảnh xám
        face_resized = cv2.resize(face_gray, (IMG_WIDTH, IMG_HEIGHT))
        
        # Chuẩn hóa và reshape lại cho đúng input của model (1, width, height, 1)
        face_for_prediction = face_resized.astype('float32') / 255.0
        face_for_prediction = np.reshape(face_for_prediction, (1, IMG_WIDTH, IMG_HEIGHT, 1))

        predictions = recognition_model.predict(face_for_prediction, verbose=0)
        
        predicted_index = np.argmax(predictions)
        predicted_name = CLASS_NAMES[predicted_index]
        confidence = np.max(predictions) * 100

        results.append({
            'box': [int(x), int(y), int(w), int(h)],
            'name': predicted_name,
            'confidence': float(f"{confidence:.2f}")
        })
        
    return results, (original_width, original_height)

# --- 4. ĐỊNH NGHĨA ROUTE (Giữ nguyên) ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify({'error': 'Không có file nào được chọn.'}), 400

    file = request.files['file']
    try:
        filename = secure_filename(str(file.filename))
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        faces, dims = detect_and_recognize_faces(filepath)

        return jsonify({'faces': faces, 'image_dimensions': dims})
    except Exception as e:
        # Ghi log lỗi chi tiết hơn để dễ debug
        import traceback
        print(f"[ERROR] Lỗi xử lý request: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Lỗi nội bộ.'}), 500

# --- 5. CHẠY SERVER ---
if __name__ == '__main__':
    if recognition_model is None or face_detector is None:
        print("\n[FATAL] Server không thể khởi động do lỗi tải mô hình.")
    else:
        print("[INFO] Server sẵn sàng tại http://127.0.0.1:5000")
        app.run(host='0.0.0.0', port=5000, debug=False)