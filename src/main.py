import streamlit as st
import cv2
import numpy as np
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from io import BytesIO
import time

# Set page configuration
st.set_page_config(page_title="Real-time Emotion Detection", layout="wide")

# Load model and resources
@st.cache_resource
def load_resources():
    # Initialize model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.load_weights('model.h5')
    
    # Load Haar cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    return model, face_cascade

def main():
    st.title("Real-time Emotion Detection with Backend")
    st.markdown("---")
    
    model, face_cascade = load_resources()
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 
                    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
    
    # Backend configuration
    backend_url = st.text_input("Backend URL", "http://192.168.1.10:8000/stream")
    
    # Create placeholder for video
    video_placeholder = st.empty()
    
    # Control buttons
    start_button = st.button("Start Detection")
    stop_button = st.button("Stop Detection")
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if start_button:
        st.session_state.processing = True
        
    if stop_button:
        st.session_state.processing = False
    
    # Initialize connection to backend
    session = requests.Session()
    
    try:
        while st.session_state.processing:
            try:
                # Get stream from backend
                response = session.get(backend_url, stream=True, timeout=5)
                
                bytes_buffer = bytes()
                for chunk in response.iter_content(chunk_size=1024):
                    if not st.session_state.processing:
                        break
                    
                    bytes_buffer += chunk
                    a = bytes_buffer.find(b'\xff\xd8')  # JPEG start
                    b = bytes_buffer.find(b'\xff\xd9')  # JPEG end
                    
                    if a != -1 and b != -1:
                        jpg = bytes_buffer[a:b+2]
                        bytes_buffer = bytes_buffer[b+2:]
                        
                        # Decode image
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if frame is not None:
                            # Convert to grayscale and detect faces
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                            
                            # Process each face
                            for (x, y, w, h) in faces:
                                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                                roi_gray = gray[y:y+h, x:x+w]
                                
                                # Prepare image for prediction
                                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                                prediction = model.predict(cropped_img)
                                maxindex = np.argmax(prediction)
                                
                                # Add emotion text
                                cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            # Convert BGR to RGB and display
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                            
            except requests.exceptions.RequestException as e:
                st.error(f"Connection error: {e}")
                st.session_state.processing = False
                
    finally:
        session.close()

if __name__ == "__main__":
    main()