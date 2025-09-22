// -----------------------------------------------------------------------------
// FILE: script.js
// MÔ TẢ: Xử lý logic phía client cho ứng dụng nhận diện khuôn mặt.
// -----------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  // --- 1. CÁC HẰNG SỐ VÀ BIẾN ---
  const CONSTANTS = {
    MAX_FILE_SIZE: 5 * 1024 * 1024, // 5MB
    ALLOWED_FILE_TYPES: ["image/jpeg", "image/png", "image/jpg"],
  };

  // Lấy các element trên trang
  const ui = {
    dropzone: document.getElementById("dropzone"),
    processingState: document.getElementById("processingState"),
    resultsState: document.getElementById("resultsState"),
    fileInput: document.getElementById("fileInput"),
    selectBtn: document.getElementById("selectBtn"),
    resetBtn: document.getElementById("resetBtn"),
    resultImage: document.getElementById("resultImage"),
    boxContainer: document.getElementById("boxContainer"),
    faceCount: document.getElementById("faceCount"),
    resultsTitle: document.getElementById("resultsTitle"),
  };

  // --- 2. QUẢN LÝ TRẠNG THÁI GIAO DIỆN ---

  /** Hiển thị một section và ẩn các section khác */
  const showSection = (sectionToShow) => {
    [ui.dropzone, ui.processingState, ui.resultsState].forEach((section) => {
      section.classList.toggle("hidden", section !== sectionToShow);
    });
  };

  // --- 3. ĐĂNG KÝ SỰ KIỆN ---

  // Mở hộp thoại chọn file
  ui.selectBtn.addEventListener("click", () => ui.fileInput.click());
  ui.dropzone.addEventListener("click", (e) => {
    if (e.target.id !== "selectBtn" && !e.target.closest("#selectBtn")) {
      ui.fileInput.click();
    }
  });

  // Xử lý khi chọn file
  ui.fileInput.addEventListener("change", (e) =>
    handleFileSelect(e.target.files)
  );

  // Xử lý kéo/thả file
  ui.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    ui.dropzone.classList.add("dragover");
  });
  ui.dropzone.addEventListener("dragleave", () =>
    ui.dropzone.classList.remove("dragover")
  );
  ui.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.dropzone.classList.remove("dragover");
    handleFileSelect(e.dataTransfer.files);
  });

  // Xử lý nút "Thử ảnh khác"
  ui.resetBtn.addEventListener("click", () => {
    ui.fileInput.value = ""; // Quan trọng: reset để có thể chọn lại file cũ
    showSection(ui.dropzone);
  });

  // --- 4. CÁC HÀM XỬ LÝ LOGIC ---

  /**
   * Kiểm tra và bắt đầu xử lý file được chọn.
   * @param {FileList} files - Danh sách file người dùng chọn.
   */
  function handleFileSelect(files) {
    if (files.length === 0) return;
    const file = files[0];

    // Kiểm tra tính hợp lệ của file
    if (file.size > CONSTANTS.MAX_FILE_SIZE) {
      alert(
        `File is too large. Maximum size is ${
          CONSTANTS.MAX_FILE_SIZE / 1024 / 1024
        }MB.`
      );
      return;
    }
    if (!CONSTANTS.ALLOWED_FILE_TYPES.includes(file.type)) {
      alert("Invalid file format. Please select a JPG or PNG image.");
      return;
    }

    // Bắt đầu quá trình xử lý
    processFile(file);
  }

  /**
   * Gửi file đến server và chờ kết quả.
   * @param {File} file - File ảnh hợp lệ.
   */
  function processFile(file) {
    showSection(ui.processingState);
    const formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok)
          throw new Error(`Server Error: ${response.statusText}`);
        return response.json();
      })
      .then((data) => {
        if (data.error) throw new Error(data.error);
        displayResults(file, data);
      })
      .catch((error) => {
        console.error("Prediction Error:", error);
        alert(`An error occurred: ${error.message}`);
        showSection(ui.dropzone); // Quay lại uploader nếu có lỗi
      });
  }

  /**
   * Hiển thị kết quả nhận diện lên giao diện.
   * @param {File} file - File ảnh gốc để hiển thị.
   * @param {object} data - Dữ liệu trả về từ server.
   */
  function displayResults(file, data) {
    const reader = new FileReader();
    reader.onload = (e) => {
      ui.resultImage.src = e.target.result;
      ui.boxContainer.innerHTML = ""; // Xóa các box cũ

      const numFaces = data.faces.length;
      ui.resultsTitle.textContent =
        numFaces > 0 ? "Detection Results" : "No Faces Detected";
      ui.faceCount.textContent =
        numFaces > 0
          ? `Found ${numFaces} face(s) in the image.`
          : "We couldn't find any faces. Please try another image.";

      const [originalWidth, originalHeight] = data.image_dimensions;

      // Vẽ các bounding box và nhãn
      data.faces.forEach((face) => {
        const [x, y, w, h] = face.box;
        const box = document.createElement("div");
        box.className = "bounding-box";

        // Dùng % để box co dãn theo kích thước ảnh
        box.style.left = `${(x / originalWidth) * 100}%`;
        box.style.top = `${(y / originalHeight) * 100}%`;
        box.style.width = `${(w / originalWidth) * 100}%`;
        box.style.height = `${(h / originalHeight) * 100}%`;

        const label = document.createElement("div");
        label.className = "face-label";
        label.textContent = `${face.name}`;

        box.appendChild(label);
        ui.boxContainer.appendChild(box);
      });

      showSection(ui.resultsState);
    };
    reader.readAsDataURL(file);
  }
});
