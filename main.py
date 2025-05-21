from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
from tensorflow.keras.models import load_model
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import socket


# Lấy địa chỉ IP của máy
def get_local_ip():
    try:
        # Tạo socket để kết nối với Google DNS
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Khởi tạo thread pool cho xử lý ảnh
thread_pool = ThreadPoolExecutor(max_workers=4)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(title="SkinVision API", description="API for skin disease analysis")

# Cấu hình CORS cho phép ứng dụng di động truy cập
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong môi trường production, nên giới hạn origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load mô hình TensorFlow
try:
    model = load_model("model/best_model.h5")
    logger.info("Model loaded successfully!")
    model.summary()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise


def process_image(image_bytes: bytes) -> np.ndarray:
    """Xử lý ảnh trong thread riêng"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((192, 192))
        img_array = np.array(image) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise


# API nhận hình ảnh và trả về kết quả phân tích
@app.post("/analyze")
async def analyze_image(file: UploadFile=File(...)):
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        # Đọc file
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        logger.info(f"File size: {len(file_bytes)} bytes")
        
        # Xử lý ảnh trong thread pool
        loop = asyncio.get_event_loop()
        img_array = await loop.run_in_executor(thread_pool, process_image, file_bytes)
        
        logger.info(f"Image array shape: {img_array.shape}")

        # Dự đoán
        try:
            prediction = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(prediction))
            logger.info(f"Predicted class: {predicted_class}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

        return {"prediction": predicted_class}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in /analyze: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "server_ip": get_local_ip()
    }


if __name__ == "__main__":
    local_ip = get_local_ip()
    logger.info(f"Server starting on IP: {local_ip}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
