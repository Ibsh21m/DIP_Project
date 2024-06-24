import cv2
from deepface import DeepFace
import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
model = load_model("mask_recog.h5")

# Function to process the image based on the selected option
def process_image(file_path, option):
    if file_path:
        # Load the image
        frame = cv2.imread(file_path)

        # Resize the image to fit within a screen resolution of 1920x1080
        max_width = 800
        max_height = 600
        height, width, _ = frame.shape
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                ratio = max_width / width
            else:
                ratio = max_height / height
            frame = cv2.resize(frame, (int(width * ratio), int(height * ratio)))

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert grayscale frame to RGB format
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # Detect faces in the frame
        faces = faceCascade.detectMultiScale(gray_frame , scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        if option == 'Face Detection':
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        elif option == 'Face Count':
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face_count = len(faces)
            cv2.putText(frame, f'Face Count: {face_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
        elif option == 'Mask Detection':
            frame = face_mask_detector(frame)
        
        elif option == 'Human Emotion Detection':
            #print(faces)
            for (x, y, w, h) in faces:
                # Extract the face ROI (Region of Interest)
                face_roi = rgb_frame[y:y + h, x:x + w]

                # Perform emotion analysis on the face ROI
                result = DeepFace.analyze(face_roi, actions=['emotion', 'race', 'age'], enforce_detection=False)

                # Determine the dominant emotion
                emotion = result[0]['dominant_emotion']
                race = result[0]['dominant_race']
                age = result[0]['age']

                # Draw rectangle around face and label with predicted emotion
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, race, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, str(age), (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


        # Convert the image to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        # Resize the image to fit the window
        img.thumbnail((800, 600))

        return ImageTk.PhotoImage(image=img)

# Function for face mask detection
def face_mask_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in faces:
        face_frame = frame[y:y + h, x:x + w]
        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_frame = cv2.resize(face_frame, (224, 224))
        face_frame = img_to_array(face_frame)
        face_frame = np.expand_dims(face_frame, axis=0)
        face_frame = preprocess_input(face_frame)
        
        preds = model.predict(face_frame)
        (mask, withoutMask) = preds[0]
        
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.1f}%".format(label, max(mask, withoutMask) * 100)
        
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
    
    return frame

# Define the layout
layout = [
    [sg.Text("Digital Image Processing Project")],
    [sg.Radio('Face Detection', "RADIO1", default=True, key='Face Detection'),
     sg.Radio('Face Count', "RADIO1", key='Face Count'),
     sg.Radio('Mask Detection', "RADIO1", key='Mask Detection'),
     sg.Radio('Human Emotion Detection', "RADIO1", key='Human Emotion Detection')],
    [sg.Button("Open Image")],
    [sg.Image(key="-IMAGE-")]
]

# Create the window
window = sg.Window("DIP Project", layout)
file_path = None
prev_option = None

# Event loop
while True:
    event, values = window.read(timeout=100)
    if event == sg.WINDOW_CLOSED:
        break
    elif event == "Open Image":
        file_path = sg.popup_get_file('Open', no_window=True)
    
    if file_path:
        option = [key for key, value in values.items() if value][0]
        if option != prev_option:
            prev_option = option
            img_tk = process_image(file_path, option)
            if img_tk:
                window["-IMAGE-"].update(data=img_tk)

# Close the window
window.close()
