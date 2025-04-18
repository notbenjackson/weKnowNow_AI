from fastapi import FastAPI, UploadFile, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import socket
import uvicorn

app = FastAPI()

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        # Create a socket connection to a public DNS server
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def print_network_info():
    """Print network information on startup"""
    local_ip = get_local_ip()
    hostname = socket.gethostname()
    
    print("\n" + "="*50)
    print(f"FastAPI Server Running!")
    print(f"Local access: http://localhost:8000")
    print(f"Network access: http://{local_ip}:8000")
    print(f"Hostname: {hostname}")
    print("="*50 + "\n")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the latest frame
latest_frame = None

@app.post("/upload")
async def receive_frame(file: bytes = Body(...)):  # Changed from UploadFile to bytes
    global latest_frame
    try:
        # Convert bytes directly to OpenCV image
        image = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        latest_frame = image
        return {"message": "Frame received successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stream")
def get_stream():
    def generate():
        global latest_frame
        while True:
            if latest_frame is not None:
                _, buffer = cv2.imencode('.jpeg', latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/ip")
def get_ip_address():
    """Endpoint to get the server IP address"""
    return {
        "local_ip": get_local_ip(),
        "hostname": socket.gethostname(),
        "access_urls": [
            f"http://localhost:8000",
            f"http://{get_local_ip()}:8000"
        ]
    }

if __name__ == "__main__":
    print_network_info()
    uvicorn.run(app, host="0.0.0.0", port=8000)