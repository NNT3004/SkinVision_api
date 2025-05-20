from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model

# Khởi tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS cho phép ứng dụng di động truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Đổi thành domain cụ thể nếu cần bảo mật
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load mô hình TensorFlow (.h5)
model = load_model("D:/React Native Code/SkinVision_api/model/best_model.h5")
print("Model loaded successfully!")
model.summary()


# API nhận hình ảnh và trả về kết quả phân tích
@app.post("/analyze")
async def analyze_image(file: UploadFile=File(...)):
    try:
        print("File received:", file.filename, file.content_type)
        file_bytes = await file.read()
        print("File size:", len(file_bytes))
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        print("Image loaded:", image.size, image.mode)

        # Resize và chuyển thành numpy array
        image = image.resize((192, 192))
        img_array = np.array(image) / 255.0  # Chuẩn hóa pixel
        img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension
        print("Image array shape:", img_array.shape)

        # Dự đoán
        prediction = model.predict(img_array)
        print("Prediction raw:", prediction)
        predicted_class = int(np.argmax(prediction))
        print("Predicted class:", predicted_class)

        return {"prediction": predicted_class}
    except Exception as e:
        print("Error in /analyze:", str(e))
        return {"error": str(e)}
