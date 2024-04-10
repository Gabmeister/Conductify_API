from flask import Flask, request, jsonify, redirect, url_for, session
from flask_cors import CORS
import requests
from urllib.parse import quote
import os
import mediapipe as mp
import numpy as np
import cv2
import keras
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)
app.secret_key = 'fypsecret2024gabriel'

# Initialize Mediapipe and fcnn model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
model = load_model('conductify_nn.h5')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['nextsong', 'pause', 'play', 'prevsong', 'volumedown', 'volumeup'])

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI")

# test route
@app.route('/')
def index():
    return 'hello world!'

@app.route('/login')
def login():
    scope = 'user-read-private user-modify-playback-state user-read-playback-state'
    auth_url = f"https://accounts.spotify.com/authorize?response_type=code&client_id={SPOTIFY_CLIENT_ID}&scope={quote(scope)}&redirect_uri={quote(SPOTIFY_REDIRECT_URI)}"
    return redirect(auth_url)

# this route is used to receive POST requests containing an auth code and exchange it for tokens
@app.route('/exchange_token', methods=['POST'])
def exchange_token():
    # get the auth code from request body
    auth_code = request.json.get('code')
    # error checking for missing code
    if not auth_code:
        return jsonify({'error': 'Missing authorization code'}), 400 
    # define Spotify API endpoint
    token_url = 'https://accounts.spotify.com/api/token'
    # set up payload for POST request to Spotify API
    data = {
        'grant_type': 'authorization_code',
        'code': auth_code,
        'redirect_uri': SPOTIFY_REDIRECT_URI,
        'client_id': SPOTIFY_CLIENT_ID,
        'client_secret': SPOTIFY_CLIENT_SECRET,
    }

    # make POST request to exchange auth code for access and refresh token
    response = requests.post(token_url, data=data)
    # error checking for missing token response
    if response.status_code != 200:
        return jsonify({'error': 'Failed to retrieve tokens', 'details': response.json()}), response.status_code
    # parse JSON response from Spotify and extract tokens
    tokens = response.json()
    access_token = tokens.get('access_token')
    refresh_token = tokens.get('refresh_token')
    # store tokens within session
    session['access_token'] = access_token
    session['refresh_token'] = refresh_token
    # return tokens in JSON format within response
    return jsonify({'message': 'Tokens exchanged successfully', 'access_token': access_token, 'refresh_token': refresh_token})


# This route is used to receive POST requests including an image and return prediction as JSON.
@app.route('/predict', methods=['POST'])
def predict_gesture():
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No file part in the request'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    original_image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    initial_results = hands.process(image_rgb) 
    prediction = 'No hands'

    if initial_results.multi_hand_landmarks:
        for hand_landmarks in initial_results.multi_hand_landmarks:
            # extract image dimensions
            h, w, _ = original_image.shape
            # get coordinates of landmarks from image
            landmark_coords = [(landmark.x * w, landmark.y * h) for landmark in hand_landmarks.landmark]
            # calculate bounding box around hand
            min_x, min_y = min(landmark_coords, key=lambda x: x[0])[0], min(landmark_coords, key=lambda x: x[1])[1]
            max_x, max_y = max(landmark_coords, key=lambda x: x[0])[0], max(landmark_coords, key=lambda x: x[1])[1]
            # crop new image using bounding box 
            cropped_image = original_image[int(min_y):int(max_y), int(min_x):int(max_x)]
            # create fixed sized image 400x400 for backdrop
            fixed_size_image = np.zeros((400, 400, 3), dtype=np.uint8)
            
            # calculate the offsets for centering cropped image onto fixed size image 
            x_offset = max((400 - cropped_image.shape[1]) // 2, 0)
            y_offset = max((400 - cropped_image.shape[0]) // 2, 0)
            fixed_size_image[y_offset:y_offset+cropped_image.shape[0], x_offset:x_offset+cropped_image.shape[1]] = cropped_image
            fixed_size_image_rgb = cv2.cvtColor(fixed_size_image, cv2.COLOR_BGR2RGB)
            
            # process new fixed size image for updated hand landmarks
            results_fixed = hands.process(fixed_size_image_rgb)
            if results_fixed.multi_hand_landmarks:
                # extract and store landmarks from fixed size image
                for fixed_hand_landmarks in results_fixed.multi_hand_landmarks:
                    normalized_landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in fixed_hand_landmarks.landmark]
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