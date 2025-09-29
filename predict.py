import tensorflow as tf
import numpy as np
import sys

# --- 1. TẢI MÔ HÌNH ĐÃ HUẤN LUYỆN ---
try:
    model = tf.keras.models.load_model('CNN_Model.h5')
    print("Đã tải xong mô hình CNN_Model.h5")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    sys.exit(1)

# --- 2. CÁC THAM SỐ ---
img_height = 64
img_width = 64
# Dựa vào thứ tự thư mục khi huấn luyện, bạn cần đảm bảo nó khớp ở đây
class_names = ['Hoan', 'Long', 'Nghia'] 
print(f"Các lớp có thể nhận diện: {class_names}")

# --- 3. HÀM TIỀN XỬ LÝ ẢNH VÀ DỰ ĐOÁN ---
def predict_image(image_path):
    """
    Hàm này nhận vào đường dẫn một ảnh, tiền xử lý và dự đoán nhãn.
    """
    try:
        # Tải ảnh từ đường dẫn
        img = tf.keras.utils.load_img(
            image_path, target_size=(img_height, img_width)
        )
        
        # Chuyển ảnh thành một mảng numpy
        img_array = tf.keras.utils.img_to_array(img)
        
        # Thêm một chiều mới để khớp với đầu vào của mô hình (batch_size, height, width, channels)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        # Thực hiện dự đoán
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        # Lấy ra nhãn và độ chính xác cao nhất
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        return predicted_class, confidence

    except FileNotFoundError:
        return f"Lỗi: Không tìm thấy file ảnh tại '{image_path}'", None
    except Exception as e:
        return f"Lỗi trong quá trình xử lý ảnh: {e}", None

# --- 4. THỰC THI DỰ ĐOÁN ---
if __name__ == "__main__":
    # --- THAY ĐỔI ĐƯỜNG DẪN ĐẾN ẢNH CỦA BẠN TẠI ĐÂY ---
    image_path_to_predict = "testLong.jpg"  
    
    predicted_class, confidence = predict_image(image_path_to_predict)

    if confidence is not None:
        print(f"\nKết quả dự đoán cho ảnh '{image_path_to_predict}':")
        print(f"  -> Khuôn mặt: {predicted_class}")
        print(f"  -> Độ tin cậy: {confidence:.2f}%")
    else:
        print(predicted_class) # In ra thông báo lỗi
