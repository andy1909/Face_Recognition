import os
import cv2
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense

# --- 1. CẤU HÌNH ---
IMG_WIDTH = 128
IMG_HEIGHT = 128
DATA_DIR = '.'
CLASS_NAMES = ['hoan', 'long', 'nghia']
NUM_CLASSES = len(CLASS_NAMES)

# --- 2. TẢI VÀ CHUẨN BỊ DỮ LIỆU ---
def load_data(data_dir, class_names):
    images = []
    labels = []
    for i, class_name in enumerate(class_names):
        path = os.path.join(data_dir, class_name, 'data_mono_rb')
        if not os.path.isdir(path):
            print(f"Cảnh báo: Thư mục không tồn tại, bỏ qua: {path}")
            continue

        print(f"Đang đọc ảnh từ thư mục: {path}")
        for filename in os.listdir(path):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(i)
    
    images = np.array(images)
    images = np.expand_dims(images, axis=-1)
    labels = np.array(labels)
    return images, labels

# Tải tất cả dữ liệu
images, labels = load_data(DATA_DIR, CLASS_NAMES)

if len(images) == 0:
    print("Lỗi: Không tìm thấy ảnh nào.")
    exit()

# Chuẩn hóa pixel và one-hot encoding
images = images.astype('float32') / 255.0
labels_cat = to_categorical(labels, NUM_CLASSES)

# --- 3. XÂY DỰNG MÔ HÌNH ---
model = Sequential()
model.add(Flatten(input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.summary()

# --- 4. HUẤN LUYỆN MÔ HÌNH ---
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nBắt đầu huấn luyện...")
# Sử dụng validation_split để Keras tự động tách 20% dữ liệu ra để kiểm thử
history = model.fit(
    images, 
    labels_cat, 
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# --- 5. ĐÁNH GIÁ VÀ LƯU MÔ HÌNH ---
print("\nHoàn tất huấn luyện.")
val_acc = history.history['val_accuracy'][-1]
print(f'Độ chính xác trên tập kiểm tra (validation): {val_acc:.4f}')

model.save('face_model.h5')
print("Đã lưu mô hình vào file 'face_model.h5'")
