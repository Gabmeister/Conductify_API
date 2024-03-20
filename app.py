from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import requests
from urllib.parse import quote
import os
import mediapipe as mp
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
app.secret_key = 'fypsecret2024gabriel'

# Initialize Mediapipe and model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
model = load_model('conductify_fcnn.keras')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['nextsong', 'pause', 'play', 'prevsong', 'volumedown', 'volumeup'])

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# Test route
@app.route('/')
def index():
    return 'hello world!'

@app.route('/login')
def login():
    scope = 'user-read-private user-modify-playback-state user-read-playback-state'
    auth_url = f"https://accounts.spotify.com/authorize?response_type=code&client_id={SPOTIFY_CLIENT_ID}&scope={quote(scope)}&redirect_uri={quote(SPOTIFY_REDIRECT_URI)}"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return jsonify({'error': 'Authorization code not found'}), 400

    auth_token_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'authorization_code',
        'code': code,
        'redirect_uri': SPOTIFY_REDIRECT_URI,
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET
    }
    
    res = requests.post(auth_token_url, data=data)
    if res.status_code != 200:
        return jsonify({'error': 'Failed to retrieve tokens', 'details': res.json()}), res.status_code

    auth_response_data = res.json()
    access_token = auth_response_data.get('access_token')
    refresh_token = auth_response_data.get('refresh_token')
    session['access_token'] = access_token
    session['refresh_token'] = refresh_token

    return jsonify({'message': 'Authentication successful', 'access_token': access_token, 'refresh_token': refresh_token})

@app.route('/exchange_token', methods=['POST'])
def exchange_token():
    # Get the authorization code from the request body
    auth_code = request.json.get('code')
    if not auth_code:
        return jsonify({'error': 'Missing authorization code'}), 400

    # token endpoint
    token_url = 'https://accounts.spotify.com/api/token'
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': SPOTIFY_REDIRECT_URI,
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET,
    }

    # Make the POST request to exchange the code for access and refresh tokens
    response = requests.post(token_url, data=data)

    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve tokens', 'details': response.json()}), response.status_code

    tokens = response.json()
    access_token = tokens.get('access_token')
    refresh_token = tokens.get('refresh_token')
    session['access_token'] = access_token
    session['refresh_token'] = refresh_token

    return jsonify({'message': 'Tokens exchanged successfully', 'access_token': access_token, 'refresh_token': refresh_token})


# This route is used to receive POST requests including an image and return prediction as JSON.
@app.route('/predict', methods=['POST'])
def predict_gesture():
    # Extract the file from the incoming request
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file part in the request'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    # Process image with Mediapipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    prediction = 'No hands'
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks from image and prepare them for prediction 
            h, w, _ = image.shape
            landmark_coords = [(landmark.x * w, landmark.y * h) for landmark in hand_landmarks.landmark]
            min_x, min_y = min(landmark_coords, key=lambda x: x[0])[0], min(landmark_coords, key=lambda x: x[1])[1]
            max_x, max_y = max(landmark_coords, key=lambda x: x[0])[0], max(landmark_coords, key=lambda x: x[1])[1]
            
            # Crop and center the image
            cropped_image = image[int(min_y):int(max_y), int(min_x):int(max_x)]
            fixed_size_image = np.zeros((400, 400, 3), dtype=np.uint8)
            x_offset = (400 - cropped_image.shape[1]) // 2
            y_offset = (400 - cropped_image.shape[0]) // 2
            fixed_size_image[y_offset:y_offset+cropped_image.shape[0], x_offset:x_offset+cropped_image.shape[1]] = cropped_image
            fixed_size_image_rgb = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2RGB)
            
            # Prepare landmarks for prediction
            normalized_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks.landmark]
            normalized_landmarks_flat = [item for sublist in normalized_landmarks for item in sublist]
            landmarks_array = np.array(normalized_landmarks_flat).reshape(1, -1)
            
            # Prediction using FCNN model
            predicted = model.predict(landmarks_array)
            predicted_label_index = np.argmax(predicted)
            predicted_label = label_encoder.inverse_transform([predicted_label_index])[0]
            prediction = predicted_label
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)