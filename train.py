import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Rescaling
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


# --- 1. CÁC THAM SỐ CẤU HÌNH ---

data_dir = '/content/drive/MyDrive/UnnameFolder/AugmentedData'

img_height = 64
img_width = 64

batch_size = 128

epochs = 10


# --- 2. TẢI VÀ CHUẨN BỊ DỮ LIỆU ---

print("Đang tải tập dữ liệu huấn luyện...")
train_ds = image_dataset_from_directory(data_dir,
                                        validation_split=0.2,
                                        subset="training",
                                        seed=42,
                                        image_size=(img_height, img_width),
                                        batch_size=batch_size)

print("\nĐang tải tập dữ liệu kiểm định...")
val_ds = image_dataset_from_directory(data_dir,
                                      validation_split=0.2,
                                      subset="validation",
                                      seed=42,
                                      image_size=(img_height, img_width),
                                      batch_size=batch_size)

# Lấy ra tên của các lớp (classes) từ tên thư mục
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"\nĐã tìm thấy các lớp (khuôn mặt): {class_names}")
print(f"Tổng số lớp: {num_classes}")


# --- 3. TỐI ƯU HÓA HIỆU SUẤT DỮ LIỆU ---
# Sử dụng cache và prefetch để tăng tốc độ đọc dữ liệu trong quá trình huấn luyện
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# --- 4. XÂY DỰNG KIẾN TRÚC MÔ HÌNH CNN ---

model = Sequential([
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    Conv2D(32, (3, 3), padding='same', activation='relu'),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    Conv2D(128, (3, 3), padding='same', activation='relu'),
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(),

    Dropout(0.5),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),

    Dense(num_classes, activation='softmax')
])


# --- 5. BIÊN DỊCH MÔ HÌNH (COMPILE) ---
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# In ra cấu trúc của mô hình
print("\nCấu trúc của mô hình CNN:")
model.summary()


# --- 6. HUẤN LUYỆN MÔ HÌNH ---
print("\nBắt đầu quá trình huấn luyện...")
history = model.fit(train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    verbose=2)
print("Hoàn tất huấn luyện!")


# --- 7. LƯU MÔ HÌNH ĐÃ HUẤN LUYỆN ---
model.save('face_recognition_model.h5')
print("\nĐã lưu mô hình đã huấn luyện vào file 'face_recognition_model.h5'")


# --- 8. ĐÁNH GIÁ VÀ VẼ BIỂU ĐỒ KẾT QUẢ ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()