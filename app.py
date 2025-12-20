import os
import re
import time
import random
import requests
import cv2
import pytesseract
import streamlink
from flask import Flask, render_template, redirect, url_for, jsonify, request
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# Set Tesseract path (adjust for your system; VS Code will use your system PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Example for Windows; comment out if on macOS/Linux

# Twitch API credentials (use environment variables for security in VS Code)
CLIENT_ID = os.environ.get('TWITCH_CLIENT_ID')
CLIENT_SECRET = os.environ.get('TWITCH_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: Twitch API credentials not set!")
    print("Please set the following environment variables:")
    print("TWITCH_CLIENT_ID=your_client_id_here")
    print("TWITCH_CLIENT_SECRET=your_client_secret_here")
    print("\nTo get these credentials:")
    print("1. Go to https://dev.twitch.tv/")
    print("2. Log in with your Twitch account")
    print("3. Go to 'Your Console' -> 'Applications'")
    print("4. Click 'Create Application'")
    print("5. Fill out the form (Name: 'Apex Stream Scanner', OAuth Redirect: 'http://localhost')")
    print("6. Copy the Client ID and Client Secret")
    print("7. Set them as environment variables before running the app")
    CLIENT_ID = None
    CLIENT_SECRET = None

app = Flask(__name__)

# GPU check removed for lightweight deployment
print("Running in lightweight mode - using CPU for OCR processing")

# Initialize Tesseract OCR (lightweight for cloud deployment)
try:
    # Test Tesseract installation
    version = pytesseract.get_tesseract_version()
    print(f"Tesseract OCR initialized: {version}")
except Exception as e:
    print(f"Tesseract not available: {e}")
    print("OCR functionality will be limited")

# Global list to hold qualifying streams
qualifying_streams = []
last_updated = ""
fetch_status = {"is_fetching": False, "progress": "", "current_stream": 0, "total_streams": 0}
# Lock for thread-safe updates
streams_lock = threading.Lock()

# Updated preprocessing variations with improved techniques
def preprocess_v1(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    binary = cv2.resize(opening, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v2(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.resize(binary, None, fx=6, fy=6, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v3(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # Invert for dark text variant
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v4(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    binary = cv2.resize(edges, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v5(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = np.ones((1,1), np.uint8)
    erode = cv2.erode(thresh, kernel, iterations=1)
    dilate = cv2.dilate(erode, kernel, iterations=1)
    binary = cv2.resize(dilate, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v6(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v7(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v8(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2,2), np.uint8)
    opening = cv2.morphologyEx(thresh[1], cv2.MORPH_OPEN, kernel)
    binary = cv2.resize(opening, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v9(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v10(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 12)
    binary = cv2.resize(binary, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v11(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 12)
    binary = cv2.resize(binary, None, fx=6, fy=6, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v12(cropped):
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=3, fy=3, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v13(cropped):  # CLAHE for contrast enhancement
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v14(cropped):  # LAB color space
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    thresh = cv2.threshold(l, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v15(cropped):  # HSV value channel
    hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v16(cropped):  # Bilateral filter
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v17(cropped):  # Gamma correction
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gamma = 0.5  # Brighten for dark areas
    adjusted = np.power(gray / 255.0, gamma) * 255.0
    adjusted = adjusted.astype(np.uint8)
    thresh = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v18(cropped):  # Top-hat morphology
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v19(cropped):  # CLAHE + adaptive threshold
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    binary = cv2.resize(binary, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v20(cropped):  # Invert + CLAHE
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

def preprocess_v21(cropped):  # LAB + bilateral filter
    lab = cv2.cvtColor(cropped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    bilateral = cv2.bilateralFilter(l, 9, 75, 75)
    thresh = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    return binary

# List of (preprocess function, language) pairs (optimized for OCR engines)
PREPROCESS_VARIANTS = [
    (preprocess_v1, 'eng'),
    (preprocess_v2, 'eng'),
    (preprocess_v3, 'eng'),
    (preprocess_v13, 'eng'),  # CLAHE
    (preprocess_v14, 'eng'),  # LAB
    (preprocess_v15, 'eng'),  # HSV value
    (preprocess_v19, 'eng'),  # CLAHE + adaptive
    (preprocess_v20, 'eng'),  # Invert + CLAHE
    (preprocess_v21, 'eng'),  # LAB + bilateral
    (preprocess_v16, 'eng'),  # Bilateral filter
    (preprocess_v17, 'eng'),  # Gamma correction
    (preprocess_v18, 'eng'),  # Top-hat morphology
]

def get_twitch_token():
    print("\n----- Fetching Twitch token -----\n")

    if not CLIENT_ID or not CLIENT_SECRET:
        print("ERROR: Twitch API credentials not set! Cannot fetch token.")
        print("Please set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET environment variables.")
        return None

    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    response = requests.post(url, params=params)
    if response.status_code != 200:
        print(f"Token error: {response.status_code} - {response.text}")
        return None
    token = response.json().get("access_token")
    print(f"Token fetched successfully: {token[:10]}...")
    return token

def get_apex_streams(token, limit=200):
    if not token:
        print("No token available, skipping streams fetch.")
        return []
    print("\n----- Fetching Apex Legends streams -----\n")
    url = "https://api.twitch.tv/helix/streams"
    headers = {
        "Authorization": f"Bearer {token}",
        "Client-Id": CLIENT_ID
    }
    all_streams = []
    cursor = None
    pages = 0
    while len(all_streams) < limit and pages < 5:  # Limit to 5 pages to avoid too many
        params = {
            "game_id": "511224",  # Apex Legends game ID
            "type": "live",
            "first": 100
        }
        if cursor:
            params["after"] = cursor

        print(f"Making API request to Twitch (page {pages + 1})...")
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Streams API error: {response.status_code} - {response.text}")
            break

        data = response.json().get("data", [])
        print(f"API returned {len(data)} streams in this page")

        if not data:
            print("No more streams in this page, stopping pagination")
            break

        # Show first few streams for debugging
        for i, stream in enumerate(data[:3]):
            print(f"  Sample stream {i+1}: {stream['user_login']} - {stream['viewer_count']} viewers - {stream['title'][:50]}...")

        streams = [{"user_login": stream["user_login"], "viewer_count": stream["viewer_count"]} for stream in data]
        all_streams.extend(streams)
        cursor = response.json().get("pagination", {}).get("cursor")
        pages += 1

        if not cursor:
            print("No cursor for next page, stopping pagination")
            break

    print(f"\nTotal fetched: {len(all_streams)} live Apex streams across {pages} pages.")

    if len(all_streams) == 0:
        print("WARNING: No Apex streams found! This could mean:")
        print("  - Game ID '511224' might be incorrect")
        print("  - No Apex streams are currently live")
        print("  - API credentials are invalid")
        print("  - Rate limiting or API issues")

        # Try to verify game ID by checking a few popular games
        print("\nTrying to verify game ID...")
        test_params = {"type": "live", "first": 5}
        test_response = requests.get(url, headers=headers, params=test_params)
        if test_response.status_code == 200:
            test_data = test_response.json().get("data", [])
            print(f"Total live streams on Twitch: {len(test_data) if test_data else 'unknown'}")
            if test_data:
                print("Sample live streams:")
                for stream in test_data[:3]:
                    print(f"  {stream['user_name']}: {stream['game_name']} ({stream['viewer_count']} viewers)")
        else:
            print("Cannot verify API - check credentials!")

    # Sort by viewer_count ascending to prioritize smaller streams, then shuffle
    all_streams.sort(key=lambda x: x["viewer_count"])
    random.shuffle(all_streams)
    streams = all_streams[:limit]  # Take up to limit random, smaller first
    print(f"Selected {len(streams)} streams for OCR processing (prioritizing smaller streams)")
    return streams

def capture_frame(username):
    print(f"Capturing frame for {username}...")
    try:
        streams = streamlink.streams(f"https://www.twitch.tv/{username}")
        if not streams:
            print(f"No streams available for {username}.")
            return None
        stream_url = streams["best"].url
        cap = cv2.VideoCapture(stream_url)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(f"Failed to read frame for {username} (ret={ret}).")
            return None
        print(f"Frame captured successfully for {username}.")
        return frame
    except Exception as e:
        print(f"Error capturing frame for {username}: {str(e)}")
        return None

def run_ocr_on_processed(binary, username, variation_idx, lang):
    squads_list = []

    # Tesseract OCR for scene text extraction
    try:
        text = pytesseract.image_to_string(binary, config='--psm 6')
        print(f"Raw OCR text Tesseract (v{variation_idx}): '{text.strip()}'")
        squads = extract_squads_from_text(text)
        if squads:
            print(f"Tesseract v{variation_idx} detected squads: {squads}")
            squads_list.extend(squads)
    except Exception as e:
        print(f"Tesseract error: {e}")

    return squads_list

def run_ocr_on_processed_kills(binary, username, variation_idx, lang):
    kills_list = []

    # Tesseract OCR for scene text extraction
    try:
        text = pytesseract.image_to_string(binary, config='--psm 6')
        print(f"Raw OCR text Tesseract kills (v{variation_idx}): '{text.strip()}'")
        kills = extract_kills_from_text(text)
        if kills:
            print(f"Tesseract v{variation_idx} detected kills: {kills}")
            kills_list.extend(kills)
    except Exception as e:
        print(f"Tesseract kills error: {e}")

    return kills_list

def extract_squads_from_text(text):
    text_lower = text.lower()
    numbers = re.findall(r'\d+', text_lower)
    squads = []
    if numbers and 'squad' in text_lower and 'left' in text_lower:
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 20:
                squads.append(num_int)
    return squads

def extract_kills_from_text(text):
    text_lower = text.lower()
    numbers = re.findall(r'\d+', text_lower)
    kills = []
    if numbers and ('kill' in text_lower or 'kills' in text_lower):
        for num in numbers:
            num_int = int(num)
            if 0 <= num_int <= 100:  # Kills can be 0-100 typically
                kills.append(num_int)
    return kills

def ocr_squad_count(frame, username):
    print("Running multi-variation OCR on frame...")
    try:
        height, width, _ = frame.shape

        # Crop for squads (top-right)
        squads_crop_x_start = int(width * 0.8)
        squads_crop_y_start = int(height * 0.01)
        squads_crop_y_end = int(height * 0.08)
        squads_cropped = frame[squads_crop_y_start:squads_crop_y_end, squads_crop_x_start:width]

        # Process squads
        all_squads = []
        for idx, (prep_func, lang) in enumerate(PREPROCESS_VARIANTS, 1):
            binary = prep_func(squads_cropped)
            squads_list = run_ocr_on_processed(binary, username, idx, lang)
            all_squads.extend(squads_list)

        # Crop for kills (top-center)
        kills_crop_x_start = int(width * 0.35)
        kills_crop_x_end = int(width * 0.65)
        kills_crop_y_start = int(height * 0.02)
        kills_crop_y_end = int(height * 0.12)
        kills_cropped = frame[kills_crop_y_start:kills_crop_y_end, kills_crop_x_start:kills_crop_x_end]

        # Process kills
        all_kills = []
        for idx, (prep_func, lang) in enumerate(PREPROCESS_VARIANTS, 1):
            binary = prep_func(kills_cropped)
            kills_list = run_ocr_on_processed_kills(binary, username, idx, lang)
            all_kills.extend(kills_list)

        # Determine squads
        squads = None
        if all_squads:
            valid_squads = [s for s in all_squads if s <= 10]
            if valid_squads:
                squads = min(valid_squads)
                print(f"Valid squads detected: {squads}")

        # Determine kills
        kills = None
        if all_kills:
            valid_kills = [k for k in all_kills if k >= 10]  # Assuming high kills start from 10+
            if valid_kills:
                kills = max(valid_kills)  # Take the highest detected kills
                print(f"Valid kills detected: {kills}")

        if squads is None and kills is None:
            print("No valid squads or kills detected.")
            return None

        return {"squads": squads, "kills": kills}
    except Exception as e:
        print(f"OCR error: {str(e)}")
        return None
    finally:
        gc.collect()

def process_stream(stream):
    username = stream["user_login"]
    print(f"\n----- Processing stream: {username} -----\n")
    frame = capture_frame(username)
    if frame is None:
        return None
    metrics = ocr_squad_count(frame, username)
    if metrics is not None:
        result = {
            "user": username,
            "squads": metrics["squads"],
            "kills": metrics["kills"],
            "url": f"https://www.twitch.tv/{username}",
            "viewers": stream["viewer_count"]
        }
        with streams_lock:
            qualifying_streams.append(result)
            # Sort by squads first, then by kills descending
            qualifying_streams.sort(key=lambda x: (x["squads"] if x["squads"] is not None else 999, -(x["kills"] if x["kills"] is not None else 0)))
        return result
    return None

def fetch_and_update_streams():
    global qualifying_streams, last_updated, fetch_status
    print("\n----- Starting stream fetch and update -----\n")

    fetch_status["progress"] = "Fetching Twitch token..."
    token = get_twitch_token()

    fetch_status["progress"] = "Getting Apex streams..."
    streams = get_apex_streams(token)

    fetch_status["total_streams"] = len(streams)
    fetch_status["progress"] = f"Processing {len(streams)} streams..."

    with streams_lock:
        qualifying_streams.clear()

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_stream, stream) for stream in streams]
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            fetch_status["current_stream"] = completed
            fetch_status["progress"] = f"Processed {completed}/{len(streams)} streams..."

    with streams_lock:
        last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
        fetch_status["is_fetching"] = False
        fetch_status["progress"] = f"Complete! Found {len(qualifying_streams)} qualifying streams."
    print(f"\n----- Update complete: Found {len(qualifying_streams)} qualifying streams. -----\n")

@app.route('/')
def home():
    return render_template('index.html', streams=qualifying_streams, last_updated=last_updated)

@app.route('/status')
def status():
    with streams_lock:
        return jsonify({
            "streams": qualifying_streams,
            "last_updated": last_updated,
            "is_fetching": fetch_status["is_fetching"],
            "progress": fetch_status["progress"],
            "current_stream": fetch_status["current_stream"],
            "total_streams": fetch_status["total_streams"]
        })

@app.route('/fetch', methods=['POST'])
def fetch():
    global fetch_status
    if fetch_status["is_fetching"]:
        return jsonify({"error": "Already fetching streams"}), 400

    fetch_status = {"is_fetching": True, "progress": "Starting...", "current_stream": 0, "total_streams": 0}
    thread = threading.Thread(target=fetch_and_update_streams)
    thread.start()
    return jsonify({"message": "Stream fetching started"})

if __name__ == '__main__':
    # For local development, optionally run initial scan
    if os.environ.get('RUN_INITIAL_SCAN', 'false').lower() == 'true':
        fetch_and_update_streams()

    # Get port from environment (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() == 'true')
