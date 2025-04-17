import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
import os
import requests
import time

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ubidots config
UBIDOTS_TOKEN = "BBUS-hdOvglR01mmpnwIbLyC9DrqyQgfK9A"  # <-- REPLACE with your actual token
UBIDOTS_DEVICE_LABEL = "esp32-sic6"
UBIDOTS_URL = f"https://industrial.api.ubidots.com/api/v1.6/devices/{UBIDOTS_DEVICE_LABEL}"

# App title
st.title("Emotion Recognition")
st.write("This app detects emotions and sends 'happy' events to Ubidots.")

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Sidebar controls
st.sidebar.header("Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, 0.4, 0.05, key="confidence_slider")
face_detection_scale = st.sidebar.slider("Face Detection Scale", 1.05, 1.5, 1.2, 0.05, key="face_scale")
face_detection_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5, 1, key="face_neighbors")
ubidots_enabled = st.sidebar.checkbox("Enable Ubidots Integration", value=True, key="ubidots_checkbox")

# Status indicators
status_placeholder = st.empty()
emotion_placeholder = st.empty()
confidence_placeholder = st.empty()
debug_placeholder = st.sidebar.empty()

# Create debug expander
debug_expander = st.sidebar.expander("Debug Information", expanded=False)
with debug_expander:
    debug_text = st.empty()

def debug_log(message):
    """Add message to debug log"""
    current_time = time.strftime("%H:%M:%S")
    if 'debug_messages' not in st.session_state:
        st.session_state.debug_messages = []
    
    st.session_state.debug_messages.append(f"{current_time}: {message}")
    # Keep only last 10 messages
    if len(st.session_state.debug_messages) > 10:
        st.session_state.debug_messages = st.session_state.debug_messages[-10:]
    
    debug_text.code('\n'.join(st.session_state.debug_messages))

def test_ubidots_connection():
    """Test the connection to Ubidots"""
    if not ubidots_enabled:
        debug_log("Ubidots integration disabled")
        return False
        
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    
    try:
        # First send a test ping
        ping_response = requests.get(
            f"https://industrial.api.ubidots.com/api/v1.6/auth/ping", 
            headers={"X-Auth-Token": UBIDOTS_TOKEN},
            timeout=5
        )
        ping_response.raise_for_status()
        debug_log(f"Ubidots ping successful: {ping_response.text}")
        
        # Then check device exists
        device_response = requests.get(
            UBIDOTS_URL,
            headers=headers,
            timeout=5
        )
        device_response.raise_for_status()
        debug_log(f"Device check successful")
        
        return True
    except Exception as e:
        debug_log(f"Ubidots test failed: {str(e)}")
        return False

def reset_ubidots():
    """Send a neutral state to Ubidots"""
    if not ubidots_enabled:
        debug_log("Ubidots integration disabled, skipping reset")
        return False
        
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    try:
        debug_log("Attempting to reset Ubidots state...")
        response = requests.post(UBIDOTS_URL, 
                     headers=headers, 
                     json={"switch": 0},
                     timeout=5)
        response.raise_for_status()
        debug_log(f"Reset success: Status {response.status_code}")
        status_placeholder.success("✅ Ubidots reset to neutral state")
        return True
    except Exception as e:
        debug_log(f"Reset failed: {str(e)}")
        status_placeholder.error(f"⚠️ Reset failed: {str(e)}")
        return False

def send_test_happy():
    """Manually send a test happy event to Ubidots"""
    debug_log("Sending test HAPPY event to Ubidots")
    result = send_emotion_to_ubidots("Happy", 1.0)
    if result:
        status_placeholder.success("✅ Test happy event sent successfully")
    else:
        status_placeholder.error("❌ Failed to send test happy event")

def send_emotion_to_ubidots(emotion, confidence):
    """Send emotion state to Ubidots"""
    if not ubidots_enabled:
        debug_log(f"Ubidots disabled, would have sent: {emotion} ({confidence:.2f})")
        return False
    
    debug_log(f"Preparing to send emotion: {emotion} with confidence {confidence:.2f}")
    
    value = 1 if emotion.lower() == "happy" else 0
    payload = {"switch": value}
    headers = {
        "X-Auth-Token": UBIDOTS_TOKEN,
        "Content-Type": "application/json"
    }
    
    try:
        debug_log(f"Sending to Ubidots: {payload}")
        response = requests.post(UBIDOTS_URL, headers=headers, json=payload, timeout=10)
        debug_log(f"Response status: {response.status_code}")
        
        response.raise_for_status()
        
        if response.status_code == 200:
            status_placeholder.success(f"✅ Sent '{emotion}' to Ubidots (Confidence: {confidence:.2f})")
            debug_log(f"Success! Response: {response.text}")
            return True
            
    except requests.exceptions.RequestException as e:
        debug_log(f"Request error: {str(e)}")
        status_placeholder.error(f"🚨 Failed to send to Ubidots: {str(e)}")
    except Exception as e:
        debug_log(f"Unexpected error: {str(e)}")
        status_placeholder.error(f"⚠️ Unexpected error: {str(e)}")
    
    return False

# Create the emotion detection model - MODIFIED: No cache decorator and model created outside
def load_model():
    """Load the emotion detection model"""
    try:
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
        
        # Load pre-trained weights
        model.load_weights('model.h5')
        debug_log("Model loaded successfully")
        return model
    except Exception as e:
        debug_log(f"Failed to load model: {str(e)}")
        return None

# Load face detection cascade
def load_face_cascade():
    """Load the face detection cascade"""
    try:
        # Try to load from OpenCV's built-in cascades
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        
        # Check if it loaded properly
        if cascade.empty():
            debug_log(f"Failed to load cascade from {cascade_path}")
            st.error(f"Failed to load cascade from {cascade_path}")
            # Try local file as fallback
            cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            if cascade.empty():
                debug_log("Failed to load local cascade file too")
                st.error("Failed to load local cascade file too")
                return None
        debug_log("Face cascade loaded successfully")
        return cascade
    except Exception as e:
        debug_log(f"Error loading face cascade: {str(e)}")
        st.error(f"Error loading face cascade: {str(e)}")
        return None

def process_image(img, face_scale=1.2, face_neighbors=5):
    """Process an image for emotion detection"""
    if img is None:
        debug_log("No image to process")
        return None
    
    debug_log(f"Processing image with shape: {img.shape}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with adjusted parameters
    faces = facecasc.detectMultiScale(
        gray, 
        scaleFactor=face_scale, 
        minNeighbors=face_neighbors,
        minSize=(30, 30)
    )
    
    debug_log(f"Detected {len(faces)} faces")
    
    # Process each detected face
    emotions_found = []
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        
        # Check if face region is valid
        if roi_gray.size > 0:
            try:
                # Resize and prepare for model
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                
                # Get emotion prediction
                prediction = model.predict(cropped_img, verbose=0)
                confidence = np.max(prediction)
                maxindex = int(np.argmax(prediction))
                detected_emotion = emotion_dict[maxindex]
                
                debug_log(f"Raw prediction: {detected_emotion} ({confidence:.2f})")
                
                # Only process emotions above confidence threshold
                if confidence >= detection_confidence:
                    emotions_found.append((detected_emotion, confidence))
                    
                    # Draw emotion text on frame
                    cv2.putText(img, f"{detected_emotion} ({confidence:.2f})", 
                              (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Update UI with detected emotion
                    emotion_placeholder.markdown(f"### Detected Emotion: **{detected_emotion}**")
                    confidence_placeholder.markdown(f"Confidence: {confidence:.2f}")
                    
                    # Send to Ubidots if happy
                    if detected_emotion.lower() == "happy":
                        debug_log(f"HAPPY detected with confidence {confidence:.2f}!")
                        send_emotion_to_ubidots(detected_emotion, confidence)
                    else:
                        debug_log(f"Not happy, resetting to neutral")
                        reset_ubidots()
            except Exception as e:
                debug_log(f"Error processing face: {str(e)}")
    
    if not emotions_found:
        debug_log("No emotions detected above threshold")
    
    # Convert back to RGB for display
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load model and cascade - MODIFIED: Load model and display error in main code
model = load_model()
if model is None:
    st.error("Could not initialize model. Please check the error messages above.")

facecasc = load_face_cascade()
if facecasc is None:
    st.error("Could not initialize face detector. Please check the error messages above.")

if model is not None and facecasc is not None:
    st.success("Model and face detector loaded successfully!")

# Ubidots test section
st.sidebar.header("Ubidots Testing")
test_col1, test_col2 = st.sidebar.columns(2)
with test_col1:
    if st.button("Test Connection"):
        if test_ubidots_connection():
            st.sidebar.success("Connection OK")
        else:
            st.sidebar.error("Connection Failed")
with test_col2:
    if st.button("Send Test Happy"):
        send_test_happy()

# Reset Ubidots at app start
if ubidots_enabled:
    reset_ubidots_button = st.sidebar.button("Reset Ubidots State", key="reset_ubidots_button")
    if reset_ubidots_button:
        reset_ubidots()

# Create a tab-based interface
tab1, tab2 = st.tabs(["Camera Mode", "Upload Mode"])

import time  # Add this at the top

with tab1:
    st.header("Camera Mode (Real-Time)")
    
    # Button to start/stop live processing
    if 'live_processing' not in st.session_state:
        st.session_state.live_processing = False
    
    if st.button("Start Live Emotion Detection"):
        st.session_state.live_processing = True
    
    if st.button("Stop Live Detection"):
        st.session_state.live_processing = False
    
    # Placeholder for the live video feed
    video_placeholder = st.empty()
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    # Track last Ubidots send time
    if 'last_ubidots_send' not in st.session_state:
        st.session_state.last_ubidots_send = 0
    
    # Process frames in a loop
    while st.session_state.live_processing:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break
        
        # Process the frame (detect emotions)
        processed_frame = process_image(frame, face_scale=face_detection_scale, face_neighbors=face_detection_neighbors)
        
        # Display the processed frame
        video_placeholder.image(processed_frame, channels="RGB")
        
        # Check if 1 second has passed since the last Ubidots send
        current_time = time.time()
        if current_time - st.session_state.last_ubidots_send >= 1:
            # If "happy" detected, send to Ubidots
            if "Happy" in st.session_state.get("last_emotion", ""):
                send_emotion_to_ubidots("Happy", st.session_state.get("last_confidence", 0.0))
                st.session_state.last_ubidots_send = current_time  # Update last send time
        
        # Small delay to prevent high CPU usage
        time.sleep(0.1)
    
    # Release camera when done
    cap.release()

with tab2:
    # Upload mode
    st.header("Upload Mode")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="file_uploader")
    
    if uploaded_file is not None:
        debug_log(f"New file uploaded: {uploaded_file.name}")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Process the image with adjusted face detection parameters
        result_img = process_image(img, face_scale=face_detection_scale, face_neighbors=face_detection_neighbors)
        
        if result_img is not None:
            # Display processed image
            st.image(result_img, caption="Processed Image", use_column_width=True)

# Reset Ubidots when app closes
if ubidots_enabled and ('was_running' not in st.session_state or st.session_state.was_running):
    st.session_state.was_running = False
    reset_ubidots()
else:
    st.session_state.was_running = True