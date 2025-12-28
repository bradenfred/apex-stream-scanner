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
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path

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

# OCR Learning System
class OCRLearningSystem:
    def __init__(self):
        self.detection_history = []
        self.preprocessing_success = {f'preprocess_v{i}': {'success': 0, 'total': 0} for i in range(1, 22)}
        self.confidence_threshold = 0.6  # Minimum confidence for "good" detection
        self.max_history = 1000  # Keep last 1000 detections

    def record_detection(self, username, preprocessing_variant, ocr_text, confidence, squads_found, kills_found):
        """Record OCR detection for learning"""
        entry = {
            'timestamp': time.time(),
            'username': username,
            'variant': preprocessing_variant,
            'text': ocr_text,
            'confidence': confidence,
            'squads_found': squads_found,
            'kills_found': kills_found,
            'successful': bool(squads_found or kills_found)
        }

        self.detection_history.append(entry)
        if len(self.detection_history) > self.max_history:
            self.detection_history.pop(0)

        # Update preprocessing success rates
        variant_key = f'preprocess_v{preprocessing_variant}'
        if variant_key in self.preprocessing_success:
            self.preprocessing_success[variant_key]['total'] += 1
            if entry['successful']:
                self.preprocessing_success[variant_key]['success'] += 1

    def get_best_variants(self, limit=5):
        """Return preprocessing variants sorted by success rate"""
        variants = []
        for variant, stats in self.preprocessing_success.items():
            if stats['total'] > 0:
                success_rate = stats['success'] / stats['total']
                variants.append((variant, success_rate, stats['total']))

        # Sort by success rate, then by total attempts
        variants.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return variants[:limit]

    def adaptive_preprocessing_order(self):
        """Reorder preprocessing variants based on learning"""
        best_variants = self.get_best_variants(10)

        # Create weighted order - prioritize successful variants
        ordered_variants = []
        for variant_name, success_rate, total in best_variants:
            # Handle both old format (preprocess_v1) and new format (combined IDs)
            if variant_name.startswith('preprocess_v'):
                variant_idx = int(variant_name.split('_v')[1])
                if variant_idx <= len(PREPROCESS_VARIANTS):
                    ordered_variants.append(PREPROCESS_VARIANTS[variant_idx - 1])
            else:
                # For combined IDs, map back to preprocessing variants
                combined_id = int(variant_name)
                prep_idx = combined_id // 10  # Extract preprocessing index
                if prep_idx <= len(PREPROCESS_VARIANTS):
                    ordered_variants.append(PREPROCESS_VARIANTS[prep_idx - 1])

        # Fill remaining with original order
        for prep_func, lang in PREPROCESS_VARIANTS:
            if (prep_func, lang) not in ordered_variants:
                ordered_variants.append((prep_func, lang))

        return ordered_variants[:8]  # Limit to 8 most effective variants for speed

    def get_detection_stats(self):
        """Return statistics about OCR performance"""
        total_detections = len(self.detection_history)
        successful_detections = sum(1 for d in self.detection_history if d['successful'])
        success_rate = successful_detections / total_detections if total_detections > 0 else 0

        return {
            'total_detections': total_detections,
            'successful_detections': successful_detections,
            'success_rate': success_rate,
            'best_variants': self.get_best_variants(3)
        }

# Global OCR learning system
ocr_learner = OCRLearningSystem()

# Training system for manual template learning
class TrainingSystem:
    def __init__(self):
        self.templates_dir = "training_templates"
        self.head_templates = []
        self.screenshot_history = []
        os.makedirs(self.templates_dir, exist_ok=True)
        self.load_templates()

    def load_templates(self):
        """Load saved templates from disk"""
        template_files = [f for f in os.listdir(self.templates_dir) if f.startswith("head_template_") and f.endswith(".png")]
        self.head_templates = []
        for template_file in template_files:
            try:
                template_path = os.path.join(self.templates_dir, template_file)
                template = cv2.imread(template_path)
                if template is not None:
                    self.head_templates.append(template)
                    print(f"Loaded template: {template_file}")
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")

    def save_head_template(self, image, x, y, w, h):
        """Save a head template from a cropped region"""
        try:
            template = image[y:y+h, x:x+w]
            if template.size > 0:
                timestamp = int(time.time())
                filename = f"head_template_{timestamp}.png"
                filepath = os.path.join(self.templates_dir, filename)
                cv2.imwrite(filepath, template)
                self.head_templates.append(template)
                print(f"Saved head template: {filename}")
                return True
        except Exception as e:
            print(f"Error saving head template: {e}")
        return False

    def get_head_templates(self):
        """Return available head templates"""
        return self.head_templates

# Global training system
training_system = TrainingSystem()

# Global list to hold qualifying streams
qualifying_streams = []
last_updated = ""
fetch_status = {"is_fetching": False, "progress": "", "current_stream": 0, "total_streams": 0, "stop_requested": False}
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

def get_apex_streams(token, limit=400):
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

    # First, verify API credentials work
    print("🔍 Verifying Twitch API credentials...")
    test_params = {"type": "live", "first": 1}
    test_response = requests.get(url, headers=headers, params=test_params)
    if test_response.status_code != 200:
        print(f"❌ API credentials invalid: {test_response.status_code} - {test_response.text}")
        print("Please check your TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET environment variables")
        return []

    print("✅ API credentials verified")

    while len(all_streams) < limit and pages < 8:  # Increased to 8 pages for better coverage
        params = {
            "game_id": "511224",  # Apex Legends game ID
            "type": "live",
            "first": 100
        }
        if cursor:
            params["after"] = cursor

        print(f"📡 Making API request to Twitch (page {pages + 1}) with params: {params}")
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ Streams API error: {response.status_code} - {response.text}")
            print(f"Request URL: {response.request.url}")
            print(f"Request headers: {dict(response.request.headers)}")

            # Try without game_id to see if that works
            if "game_id" in params:
                print("🔄 Trying without game_id filter...")
                fallback_params = {"type": "live", "first": 10}
                fallback_response = requests.get(url, headers=headers, params=fallback_params)
                if fallback_response.status_code == 200:
                    fallback_data = fallback_response.json().get("data", [])
                    print(f"✅ API works without game_id, found {len(fallback_data)} total streams")
                    # Filter for Apex streams manually
                    apex_streams = [s for s in fallback_data if s.get("game_id") == "511224"]
                    if apex_streams:
                        print(f"🎯 Found {len(apex_streams)} Apex streams by filtering")
                        streams = [{"user_login": stream["user_login"], "viewer_count": stream["viewer_count"]} for stream in apex_streams[:limit]]
                        return streams[:limit]
                else:
                    print(f"❌ API completely broken: {fallback_response.status_code}")
            break

        data = response.json().get("data", [])
        print(f"📊 API returned {len(data)} streams in this page")

        if not data:
            print("📭 No more streams in this page, stopping pagination")
            break

        # Show first few streams for debugging
        for i, stream in enumerate(data[:3]):
            game_name = stream.get("game_name", "Unknown")
            print(f"  🎮 Sample stream {i+1}: {stream['user_login']} - {stream['viewer_count']} viewers - {game_name} - {stream['title'][:50]}...")

        streams = [{"user_login": stream["user_login"], "viewer_count": stream["viewer_count"]} for stream in data]
        all_streams.extend(streams)
        cursor = response.json().get("pagination", {}).get("cursor")
        pages += 1

        if not cursor:
            print("🔚 No cursor for next page, stopping pagination")
            break

    print(f"\n🎯 Total fetched: {len(all_streams)} live Apex streams across {pages} pages.")

    if len(all_streams) == 0:
        print("⚠️ WARNING: No Apex streams found! This could mean:")
        print("  - Game ID '511224' might be incorrect or changed")
        print("  - No Apex streams are currently live")
        print("  - API rate limiting")
        print("  - Game might not be categorized properly")

        # Try to find the correct game ID
        print("\n🔍 Attempting to find Apex Legends game ID...")
        games_url = "https://api.twitch.tv/helix/games"
        games_params = {"name": "Apex Legends"}
        games_response = requests.get(games_url, headers=headers, params=games_params)

        if games_response.status_code == 200:
            games_data = games_response.json().get("data", [])
            if games_data:
                correct_game_id = games_data[0]["id"]
                correct_game_name = games_data[0]["name"]
                print(f"🎯 Found Apex Legends - ID: {correct_game_id}, Name: {correct_game_name}")
                if correct_game_id != "511224":
                    print(f"⚠️ Game ID mismatch! Using {correct_game_id} instead of 511224")
                    # Try one more time with correct ID
                    retry_params = {
                        "game_id": correct_game_id,
                        "type": "live",
                        "first": 20
                    }
                    retry_response = requests.get(url, headers=headers, params=retry_params)
                    if retry_response.status_code == 200:
                        retry_data = retry_response.json().get("data", [])
                        if retry_data:
                            print(f"✅ Found {len(retry_data)} streams with correct game ID")
                            streams = [{"user_login": stream["user_login"], "viewer_count": stream["viewer_count"]} for stream in retry_data]
                            return streams[:limit]
            else:
                print("❌ Could not find Apex Legends in games database")
        else:
            print(f"❌ Games API error: {games_response.status_code}")

    # Sort by viewer_count ascending to prioritize smaller streams, then shuffle
    all_streams.sort(key=lambda x: x["viewer_count"])
    random.shuffle(all_streams)
    streams = all_streams[:limit]  # Take up to limit random, smaller first
    print(f"🎲 Selected {len(streams)} streams for OCR processing (prioritizing smaller streams)")
    return streams

def check_stream_health(username, timeout_seconds=2):
    """Quick health check to see if stream is accessible and active"""
    print(f"🏥 Health check for {username} (timeout: {timeout_seconds}s)...")
    try:
        import threading
        result = [None]
        error = [None]

        def health_worker():
            try:
                # Quick check if stream exists and is accessible
                streams = streamlink.streams(f"https://www.twitch.tv/{username}")
                if not streams:
                    error[0] = f"No streams available for {username}."
                    return

                # Try to open the stream briefly to verify it's working
                stream_url = streams["best"].url
                cap = cv2.VideoCapture(stream_url)

                # Quick read attempt (don't actually read frame yet)
                if not cap.isOpened():
                    error[0] = f"Cannot open stream for {username}."
                    cap.release()
                    return

                # Check if we can get basic properties
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

                if width == 0 or height == 0:
                    error[0] = f"Invalid stream dimensions for {username}."
                    cap.release()
                    return

                cap.release()
                result[0] = True
                print(f"✅ Stream {username} is healthy")

            except Exception as e:
                error[0] = f"Health check failed for {username}: {str(e)}"

        # Start health check in background thread
        health_thread = threading.Thread(target=health_worker, daemon=True)
        health_thread.start()

        # Wait for completion with short timeout
        health_thread.join(timeout=timeout_seconds)

        if health_thread.is_alive():
            print(f"⏰ Health check timeout for {username} - stream may be slow")
            return False  # Consider slow streams unhealthy for quick scanning

        if error[0]:
            print(f"❌ Health check failed for {username}: {error[0]}")
            return False

        return result[0] is True

    except Exception as e:
        print(f"Unexpected error in health check for {username}: {str(e)}")
        return False

def capture_frame(username, timeout_seconds=5):
    print(f"📸 Capturing frame for {username} (timeout: {timeout_seconds}s)...")
    try:
        import threading
        result = [None]
        error = [None]

        def capture_worker():
            try:
                streams = streamlink.streams(f"https://www.twitch.tv/{username}")
                if not streams:
                    error[0] = f"No streams available for {username}."
                    return

                stream_url = streams["best"].url
                cap = cv2.VideoCapture(stream_url)
                ret, frame = cap.read()
                cap.release()

                if not ret:
                    error[0] = f"Failed to read frame for {username} (ret={ret})."
                    return

                result[0] = frame
                print(f"✅ Frame captured successfully for {username}.")

            except Exception as e:
                error[0] = f"Error capturing frame for {username}: {str(e)}"

        # Start capture in background thread
        capture_thread = threading.Thread(target=capture_worker, daemon=True)
        capture_thread.start()

        # Wait for completion with timeout
        capture_thread.join(timeout=timeout_seconds)

        if capture_thread.is_alive():
            print(f"⏰ Timeout: Capture for {username} took longer than {timeout_seconds} seconds, skipping...")
            return None

        if error[0]:
            print(f"❌ Capture failed for {username}: {error[0]}")
            return None

        return result[0]

    except Exception as e:
        print(f"Unexpected error in capture_frame for {username}: {str(e)}")
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
    """Extract squads remaining from OCR text with Apex-specific patterns"""
    text_lower = text.lower()
    numbers = re.findall(r'\d+', text_lower)

    squads = []
    # Apex Legends specific patterns - expanded for better detection
    apex_squad_patterns = [
        r'(\d+)/20',                    # "3/20" squads
        r'(\d+)\s*squads?\s*left',      # "3 squads left"
        r'(\d+)\s*squads?\s*remaining', # "3 squads remaining"
        r'squads?:\s*(\d+)',           # "Squads: 3"
        r'(\d+)\s*alive',               # "3 alive"
        r'(\d+)\s*teams?\s*left',       # "3 teams left"
        r'(\d+)\s*of\s*20',             # "3 of 20"
        r'(\d+)\s*left',                # "3 left"
        r'remaining:\s*(\d+)',          # "Remaining: 3"
        r'(\d+)\s*squads?',             # "3 squads" (simple)
        r'(\d+)\s*teams?',              # "3 teams" (simple)
        r'(\d+)\s*sq',                  # "3 sq" (abbreviated)
        r'sq\s*(\d+)',                  # "sq 3" (abbreviated)
        r'(\d+)[\s/]+(?:sq|squads?|teams?)',  # "3 sq", "3/squads"
    ]

    # Generic fallback patterns
    generic_squad_patterns = [
        r'(\d+)\s*squad',  # "3 squad"
        r'squad\s*(\d+)',  # "squad 3"
        r'(\d+)\s*remaining', # "3 remaining"
        r'remaining\s*(\d+)', # "remaining 3"
        r'(\d+)\s*active',   # "3 active"
        r'active\s*(\d+)',   # "active 3"
    ]

    # Try Apex patterns first
    for pattern in apex_squad_patterns + generic_squad_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            num_int = int(match)
            if 1 <= num_int <= 20:
                squads.append(num_int)

    # Special handling for "X/20" format
    slash_matches = re.findall(r'(\d+)/20', text_lower)
    for match in slash_matches:
        num_int = int(match)
        if 1 <= num_int <= 20:
            squads.append(num_int)

    # Context-based extraction for unclear cases - improved logic
    squad_keywords = ['squad', 'team', 'sq', 'alive', 'remaining', 'left', 'active']
    has_squad_context = any(keyword in text_lower for keyword in squad_keywords)

    if has_squad_context:
        # Look for isolated numbers that could be squad counts
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 20:
                # Additional validation: check if number appears near squad keywords
                num_str = str(num_int)
                start_pos = text_lower.find(num_str)
                if start_pos >= 0:
                    # Check 10 characters before and after for context
                    context_before = text_lower[max(0, start_pos-10):start_pos]
                    context_after = text_lower[start_pos+len(num_str):start_pos+len(num_str)+10]
                    full_context = context_before + ' ' + context_after

                    if any(keyword in full_context for keyword in squad_keywords):
                        squads.append(num_int)
                    # Also accept if it's a reasonable squad number and we have squad context
                    elif 1 <= num_int <= 10:  # Endgame numbers are more likely
                        squads.append(num_int)

    return list(set(squads))  # Remove duplicates

def extract_kills_from_text(text):
    """Extract kills from OCR text - but also prepare for visual detection"""
    text_lower = text.lower()
    numbers = re.findall(r'\d+', text_lower)
    kills = []

    # Text-based kill detection (less common)
    kill_patterns = [
        r'(\d+)\s*kill',   # "5 kill"
        r'kill\s*(\d+)',   # "kill 5"
        r'(\d+)\s*kills',  # "5 kills"
        r'kills\s*(\d+)',  # "kills 5"
    ]

    for pattern in kill_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            num_int = int(match)
            if 0 <= num_int <= 100:
                kills.append(num_int)

    return list(set(kills))  # Remove duplicates

def detect_kills_visual(image, head_template=None):
    """Detect kills using visual template matching for head silhouettes"""
    if head_template is None:
        # For now, return None - will implement template loading later
        return None

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template_gray = cv2.cvtColor(head_template, cv2.COLOR_BGR2GRAY)

        # Template matching
        result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
        threshold = 0.7  # Similarity threshold

        # Find locations above threshold
        locations = np.where(result >= threshold)
        kill_count = len(locations[0]) if locations[0].size > 0 else 0

        return kill_count if kill_count > 0 else None

    except Exception as e:
        print(f"Visual kill detection error: {e}")
        return None

def ocr_squad_count(frame, username):
    global processing_details, custom_roi_positions
    print("🎯 Running APEX-optimized OCR with multiple PSM modes...")
    try:
        height, width, _ = frame.shape

        # Use custom ROI positions if available, otherwise use defaults
        if custom_roi_positions:
            print("🎯 Using custom ROI positions")
            kills_roi = custom_roi_positions['kills']
            squads_roi = custom_roi_positions['squads']

            # Convert relative positions (0-1) to absolute pixel coordinates
            kills_x = int(kills_roi['x'] * width)
            kills_y = int(kills_roi['y'] * height)
            kills_w = int(kills_roi['width'] * width)
            kills_h = int(kills_roi['height'] * height)

            squads_x = int(squads_roi['x'] * width)
            squads_y = int(squads_roi['y'] * height)
            squads_w = int(squads_roi['width'] * width)
            squads_h = int(squads_roi['height'] * height)
        else:
            print("🎯 Using default ROI positions")
            # Improved APEX HUD CROPS: Squads in bottom-center, Kills in top-center
            # Squads are typically in the center-bottom of the screen
            squads_x = int(width * 0.40)  # More center-left for squads
            squads_y = int(height * 0.88)  # Lower on screen
            squads_w = int(width * 0.20)   # Wider area
            squads_h = int(height * 0.08)  # Taller for text

            # Kills are typically in the top-center area
            kills_x = int(width * 0.35)   # Center area
            kills_y = int(height * 0.02)  # Very top
            kills_w = int(width * 0.30)   # Wide area for kill feed
            kills_h = int(height * 0.08)  # Reasonable height

        # Extract ROI regions
        kills_crop = frame[kills_y:kills_y + kills_h, kills_x:kills_x + kills_w]
        squads_crop = frame[squads_y:squads_y + squads_h, squads_x:squads_x + squads_w]

        print(f"📐 Kills ROI: ({kills_x}, {kills_y}) {kills_w}x{kills_h}")
        print(f"📐 Squads ROI: ({squads_x}, {squads_y}) {squads_w}x{squads_h}")

        # FAST PATH: Try visual kill detection first (much faster than OCR)
        kills_from_visual = None
        if training_system.get_head_templates():
            for head_template in training_system.get_head_templates():
                visual_kills = detect_kills_visual(kills_crop, head_template)
                if visual_kills is not None and visual_kills > 0:
                    kills_from_visual = visual_kills
                    print(f"⚡ FAST VISUAL: Found {visual_kills} kills using template")
                    break

        # Use adaptive preprocessing order (limited to 3 best variants for speed)
        preprocessing_order = ocr_learner.adaptive_preprocessing_order()[:3]

        # MULTIPLE PSM MODES for different text layouts
        psm_modes = ['--psm 6', '--psm 7', '--psm 8', '--psm 11', '--psm 13']

        all_squads = []
        all_kills = []
        variant_results = []
        best_squads_text = ""
        best_kills_text = ""

        def process_ocr_variant(variant_info):
            """Process OCR with different preprocessing + PSM combinations"""
            prep_idx, (prep_func, lang) = variant_info

            results = []
            for psm_idx, psm_config in enumerate(psm_modes[:3]):  # Limit PSM modes for speed
                try:
                    # Process squads region
                    squads_binary = prep_func(squads_crop)
                    squads_ocr_data = pytesseract.image_to_data(squads_binary, config=psm_config, output_type=pytesseract.Output.DICT)
                    squads_text = ' '.join([word for word in squads_ocr_data['text'] if word.strip()])

                    # Process kills region
                    kills_binary = prep_func(kills_crop)
                    kills_ocr_data = pytesseract.image_to_data(kills_binary, config=psm_config, output_type=pytesseract.Output.DICT)
                    kills_text = ' '.join([word for word in kills_ocr_data['text'] if word.strip()])

                    # Combine confidences
                    all_confidences = squads_ocr_data['conf'] + kills_ocr_data['conf']
                    valid_confidences = [c for c in all_confidences if c >= 0]
                    avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0

                    # Extract data from both regions
                    squads_from_text = extract_squads_from_text(squads_text)
                    kills_from_text = extract_kills_from_text(kills_text)

                    # Combine text and visual kill detections
                    all_kills_from_variant = kills_from_text[:]
                    if kills_from_visual is not None:
                        all_kills_from_variant.append(kills_from_visual)

                    successful = bool(squads_from_text or all_kills_from_variant)

                    results.append({
                        'prep_variant': prep_idx,
                        'psm_mode': psm_idx + 6,  # PSM mode number
                        'squads_text': squads_text,
                        'kills_text': kills_text,
                        'combined_text': f"{squads_text} | {kills_text}",
                        'confidence': avg_confidence,
                        'squads': squads_from_text,
                        'kills': all_kills_from_variant,
                        'successful': successful
                    })

                except Exception as e:
                    print(f"OCR error prep_v{prep_idx} psm_{psm_idx + 6}: {e}")

            return results

        # Process preprocessing variants in parallel
        variant_infos = [(idx, prep_info) for idx, prep_info in enumerate(preprocessing_order, 1)]

        with ThreadPoolExecutor(max_workers=min(3, len(variant_infos))) as variant_executor:
            variant_futures = [variant_executor.submit(process_ocr_variant, info) for info in variant_infos]

            for future in as_completed(variant_futures):
                results = future.result()
                for result in results:
                    if result:
                        variant_results.append(result)

                        # Store best text results for monitoring
                        if result['squads_text'] and len(result['squads_text']) > len(best_squads_text):
                            best_squads_text = result['squads_text']
                        if result['kills_text'] and len(result['kills_text']) > len(best_kills_text):
                            best_kills_text = result['kills_text']

                        # Print results for debugging
                        print(f"🎯 APEX OCR prep_v{result['prep_variant']} psm_{result['psm_mode']}: '{result['combined_text'][:50]}...' (conf: {result['confidence']:.1f}%)")

                        if result['squads']:
                            print(f"  → Squads: {result['squads']}")
                            all_squads.extend(result['squads'])

                        if result['kills']:
                            print(f"  → Kills: {result['kills']}")
                            all_kills.extend(result['kills'])

                        # Record this detection for learning (use combined variant ID)
                        combined_variant = result['prep_variant'] * 10 + result['psm_mode']
                        ocr_learner.record_detection(
                            username=username,
                            preprocessing_variant=combined_variant,
                            ocr_text=result['combined_text'],
                            confidence=result['confidence'] / 100.0,
                            squads_found=result['squads'],
                            kills_found=result['kills']
                        )

                        # Early exit if we found good results with high confidence
                        if result['successful'] and result['confidence'] > 60:  # Lower threshold for training
                            print(f"🎯 High-confidence APEX detection found, continuing...")
                            break

        # IMPROVED ENSEMBLE VOTING with single high-confidence acceptance
        final_squads = None
        final_kills = None

        if all_squads:
            squad_counts = {}
            for squad in all_squads:
                squad_counts[squad] = squad_counts.get(squad, 0) + 1

            max_count = max(squad_counts.values())
            if max_count >= 2:  # At least 2 variants agreed
                final_squads = [squad for squad, count in squad_counts.items() if count == max_count][0]
            elif max_count == 1 and len(variant_results) > 0:  # Accept single detection if we have results
                # Take the most reasonable squad count
                valid_squads = [s for s in all_squads if 1 <= s <= 20]
                if valid_squads:
                    # Prioritize endgame squads (lower numbers)
                    final_squads = min(valid_squads) if min(valid_squads) <= 10 else max(valid_squads)

        if all_kills:
            kill_counts = {}
            for kill in all_kills:
                kill_counts[kill] = kill_counts.get(kill, 0) + 1

            max_count = max(kill_counts.values())
            if max_count >= 2:  # At least 2 variants agreed
                final_kills = [kill for kill, count in kill_counts.items() if count == max_count][0]
            elif max_count == 1 and len(variant_results) > 0:  # Accept single detection
                valid_kills = [k for k in all_kills if k >= 0]
                if valid_kills:
                    final_kills = max(valid_kills)  # Take highest kill count

        # STORE OCR TEXT RESULTS FOR MONITORING
        processing_details["ocr_results"]["squads_text"] = best_squads_text
        processing_details["ocr_results"]["kills_text"] = best_kills_text

        # Print learning stats occasionally
        if len(ocr_learner.detection_history) % 25 == 0 and len(ocr_learner.detection_history) > 0:  # More frequent stats
            stats = ocr_learner.get_detection_stats()
            print(f"\n--- APEX OCR Learning Stats ---")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
            print(f"Best variants: {stats['best_variants']}")
            print("-------------------------------\n")

        if final_squads is not None:
            print(f"✅ Final APEX squads detected: {final_squads}")
        if final_kills is not None:
            print(f"✅ Final APEX kills detected: {final_kills}")

        if final_squads is None and final_kills is None:
            print("❌ No valid APEX squads or kills detected")
            return None

        return {"squads": final_squads, "kills": final_kills}

    except Exception as e:
        print(f"❌ APEX OCR error: {str(e)}")
        return None
    finally:
        gc.collect()

def process_stream(stream):
    global processing_details
    username = stream["user_login"]
    print(f"\n----- Processing stream: {username} -----\n")

    # UPDATE MONITORING: Set current stream being processed
    processing_details["current_stream"] = {
        "username": username,
        "viewers": stream["viewer_count"],
        "url": f"https://www.twitch.tv/{username}"
    }
    processing_details["status"] = "processing"
    processing_details["timeline"] = []
    processing_details["processing_log"] = []
    start_time = time.time()

    def add_log_entry(message):
        """Add entry to processing timeline and log"""
        elapsed = time.time() - start_time
        entry = ".1f"
        processing_details["timeline"].append(entry)
        processing_details["processing_log"].append(message)
        print(message)

    add_log_entry("Starting stream processing")

    # HEALTH CHECK: Verify stream is accessible before processing
    add_log_entry("Health check started")
    if not check_stream_health(username, timeout_seconds=8):
        add_log_entry(f"❌ Health check failed - stream unhealthy")
        processing_details["status"] = "failed"
        print(f"⏭️ Skipping unhealthy stream: {username}")
        return None
    add_log_entry("✅ Health check passed")

    add_log_entry("Frame capture started")
    frame = None
    max_retries = 2
    for attempt in range(max_retries + 1):
        if attempt > 0:
            add_log_entry(f"Retrying frame capture (attempt {attempt + 1}/{max_retries + 1})")
            time.sleep(1)  # Brief pause between retries

        frame = capture_frame(username, timeout_seconds=7)  # Slightly shorter timeout for retries
        if frame is not None:
            break

        if attempt < max_retries:
            print(f"⚠️ Frame capture attempt {attempt + 1} failed, retrying...")

    if frame is None:
        add_log_entry("❌ Frame capture failed after retries")
        processing_details["status"] = "failed"
        print(f"❌ Failed to capture frame from {username} after {max_retries + 1} attempts")
        return None
    add_log_entry("✅ Frame captured")

    # STORE SCREENSHOT FOR MONITORING
    import base64
    _, buffer = cv2.imencode('.png', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    processing_details["screenshot"] = f"data:image/png;base64,{img_base64}"

    # CALCULATE ROI REGIONS FOR MONITORING (same logic as in ocr_squad_count)
    height, width = frame.shape[:2]

    # Use custom ROI positions if available, otherwise use defaults
    if custom_roi_positions:
        print("🎯 Using custom ROI positions for monitoring")
        kills_roi = custom_roi_positions['kills']
        squads_roi = custom_roi_positions['squads']

        # Convert relative positions (0-1) to absolute pixel coordinates
        monitor_kills_x = int(kills_roi['x'] * width)
        monitor_kills_y = int(kills_roi['y'] * height)
        monitor_kills_w = int(kills_roi['width'] * width)
        monitor_kills_h = int(kills_roi['height'] * height)

        monitor_squads_x = int(squads_roi['x'] * width)
        monitor_squads_y = int(squads_roi['y'] * height)
        monitor_squads_w = int(squads_roi['width'] * width)
        monitor_squads_h = int(squads_roi['height'] * height)
    else:
        print("🎯 Using default ROI positions for monitoring")
        # Improved APEX HUD CROPS: Squads in bottom-center, Kills in top-center
        # Squads are typically in the center-bottom of the screen
        monitor_squads_x = int(width * 0.40)  # More center-left for squads
        monitor_squads_y = int(height * 0.88)  # Lower on screen
        monitor_squads_w = int(width * 0.20)   # Wider area
        monitor_squads_h = int(height * 0.08)  # Taller for text

        # Kills are typically in the top-center area
        monitor_kills_x = int(width * 0.35)   # Center area
        monitor_kills_y = int(height * 0.02)  # Very top
        monitor_kills_w = int(width * 0.30)   # Wide area for kill feed
        monitor_kills_h = int(height * 0.08)  # Reasonable height

    processing_details["roi_regions"] = {
        "kills": {
            "x": monitor_kills_x,
            "y": monitor_kills_y,
            "width": monitor_kills_w,
            "height": monitor_kills_h
        },
        "squads": {
            "x": monitor_squads_x,
            "y": monitor_squads_y,
            "width": monitor_squads_w,
            "height": monitor_squads_h
        }
    }

    add_log_entry("OCR processing started")
    metrics = ocr_squad_count(frame, username)

    if metrics is not None:
        # UPDATE OCR RESULTS FOR MONITORING
        processing_details["ocr_results"] = {
            "kills_text": getattr(ocr_squad_count, 'last_kills_text', ''),
            "squads_text": getattr(ocr_squad_count, 'last_squads_text', ''),
            "kills_detected": metrics["kills"],
            "squads_detected": metrics["squads"],
            "confidence": {"kills": 85, "squads": 90}  # Placeholder confidence
        }

        result = {
            "user": username,
            "squads": metrics["squads"],
            "kills": metrics["kills"],
            "url": f"https://www.twitch.tv/{username}",
            "viewers": stream["viewer_count"]
        }

        add_log_entry(f"🎯 OCR detection complete - Squads: {metrics['squads']}, Kills: {metrics['kills']}")
        processing_details["status"] = "completed"

        # IMMEDIATELY add to global results for real-time updates
        with streams_lock:
            qualifying_streams.append(result)
            # Sort by squads first, then by kills descending
            qualifying_streams.sort(key=lambda x: (x["squads"] if x["squads"] is not None else 999, -(x["kills"] if x["kills"] is not None else 0)))
            print(f"✅ REAL-TIME UPDATE: Added {username} to results (total: {len(qualifying_streams)} streams)")
        return result

    add_log_entry("❌ No qualifying content found")
    processing_details["status"] = "no_content"
    print(f"❌ No qualifying content found in {username}")
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

    # DON'T CLEAR RESULTS - maintain them for real-time updates
    # Only clear if this is a fresh scan (not auto-refresh)
    if not hasattr(fetch_and_update_streams, 'last_scan_time') or \
       time.time() - fetch_and_update_streams.last_scan_time > 300:  # 5 minutes
        print("🔄 Fresh scan - clearing old results")
        with streams_lock:
            qualifying_streams.clear()
    else:
        print("🔄 Auto-refresh - keeping existing results")

    fetch_and_update_streams.last_scan_time = time.time()

    # Process streams SEQUENTIALLY (one at a time) for clearer real-time results
    completed = 0
    for stream in streams:
        # Check if stop was requested
        if fetch_status["stop_requested"]:
            print("Stop signal received, halting processing...")
            fetch_status["progress"] = f"Stopped! Processed {completed}/{len(streams)} streams."
            break

        # Process this stream
        result = process_stream(stream)
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

    # Check if we're already fetching
    if fetch_status["is_fetching"]:
        return jsonify({"error": "Already fetching streams"}), 400

    # Validate Twitch credentials before starting
    if not CLIENT_ID or not CLIENT_SECRET:
        error_msg = "Twitch API credentials not configured. Please set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET environment variables."
        print(f"❌ FETCH ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 400

    # Test Twitch API connectivity
    print("🔍 Testing Twitch API connection before starting scan...")
    test_token = get_twitch_token()
    if not test_token:
        error_msg = "Failed to obtain Twitch API token. Check your credentials."
        print(f"❌ FETCH ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 400

    # Test API with a simple request
    test_url = "https://api.twitch.tv/helix/streams"
    test_headers = {
        "Authorization": f"Bearer {test_token}",
        "Client-Id": CLIENT_ID
    }
    test_response = requests.get(test_url, headers=test_headers, params={"type": "live", "first": 1})

    if test_response.status_code != 200:
        error_msg = f"Twitch API test failed: HTTP {test_response.status_code} - {test_response.text}"
        print(f"❌ FETCH ERROR: {error_msg}")
        return jsonify({"error": error_msg}), 400

    print("✅ Twitch API connection verified")

    # Start the fetch process - preserve stop_requested flag
    stop_requested = fetch_status.get("stop_requested", False)
    fetch_status = {"is_fetching": True, "progress": "Starting...", "current_stream": 0, "total_streams": 0, "stop_requested": stop_requested}
    thread = threading.Thread(target=fetch_and_update_streams)
    thread.start()
    return jsonify({"message": "Stream fetching started"})

@app.route('/ocr-stats')
def ocr_stats():
    """Return OCR learning statistics"""
    with streams_lock:
        stats = ocr_learner.get_detection_stats()
        return jsonify({
            "ocr_learning_stats": stats,
            "streams_analyzed": len(qualifying_streams),
            "last_updated": last_updated
        })

# Global processing details for monitoring
processing_details = {
    "current_stream": None,
    "status": "idle",
    "screenshot": None,
    "roi_regions": {
        "kills": {"x": 0, "y": 0, "width": 0, "height": 0},
        "squads": {"x": 0, "y": 0, "width": 0, "height": 0}
    },
    "ocr_results": {
        "kills_text": "",
        "squads_text": "",
        "kills_detected": None,
        "squads_detected": None,
        "confidence": {"kills": 0, "squads": 0}
    },
    "timeline": [],
    "processing_log": []
}

@app.route('/processing-details')
def processing_details_endpoint():
    """Return current processing details for monitoring"""
    global processing_details
    return jsonify(processing_details)

@app.route('/save-roi-positions', methods=['POST'])
def save_roi_positions():
    """Save custom ROI positions for OCR processing"""
    try:
        data = request.get_json()
        kills_roi = data.get('kills_roi')
        squads_roi = data.get('squads_roi')

        if not kills_roi or not squads_roi:
            return jsonify({"error": "Missing ROI data"}), 400

        # Store custom ROIs globally for use in OCR processing
        global custom_roi_positions
        custom_roi_positions = {
            'kills': kills_roi,
            'squads': squads_roi
        }

        print(f"💾 Saved custom ROI positions: Kills={kills_roi}, Squads={squads_roi}")
        return jsonify({"success": True, "message": "ROI positions saved"})

    except Exception as e:
        print(f"❌ Error saving ROI positions: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/reset-roi-positions', methods=['POST'])
def reset_roi_positions():
    """Reset ROI positions to defaults"""
    try:
        global custom_roi_positions
        custom_roi_positions = None
        print("🔄 ROI positions reset to defaults")
        return jsonify({"success": True, "message": "ROI positions reset"})

    except Exception as e:
        print(f"❌ Error resetting ROI positions: {e}")
        return jsonify({"error": str(e)}), 500

# Global custom ROI positions
custom_roi_positions = None

def cleanup_temp_screenshots():
    """Clean up temporary training screenshots older than 1 hour"""
    try:
        import glob
        current_time = time.time()

        # Find all temporary screenshots
        temp_files = glob.glob("training_screenshot_*.png")

        cleaned_count = 0
        for filepath in temp_files:
            try:
                # Check file age
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > 3600:  # 1 hour
                    os.remove(filepath)
                    cleaned_count += 1
                    print(f"🧹 Cleaned up old training screenshot: {filepath}")
            except Exception as e:
                print(f"Error cleaning up {filepath}: {e}")

        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} temporary training screenshots")

    except Exception as e:
        print(f"Error during cleanup: {e}")

@app.route('/capture-screenshot/<username>')
def capture_screenshot(username):
    """Capture a screenshot from a specific stream for training"""
    try:
        # Clean up old screenshots before creating new ones
        cleanup_temp_screenshots()

        frame = capture_frame(username)
        if frame is None:
            return jsonify({"error": "Failed to capture frame from stream"}), 400

        # Save screenshot temporarily for training
        timestamp = int(time.time())
        screenshot_path = f"training_screenshot_{username}_{timestamp}.png"
        cv2.imwrite(screenshot_path, frame)

        # Return screenshot data
        import base64
        _, buffer = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            "success": True,
            "screenshot": f"data:image/png;base64,{img_base64}",
            "username": username,
            "timestamp": timestamp,
            "dimensions": {"width": frame.shape[1], "height": frame.shape[0]}
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save-template', methods=['POST'])
def save_template():
    """Save a template from training data"""
    try:
        data = request.get_json()
        username = data.get('username')
        template_type = data.get('type')  # 'head' for now
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        width = int(data.get('width', 0))
        height = int(data.get('height', 0))
        timestamp = data.get('timestamp')

        print(f"📝 Saving template - Username: {username}, Type: {template_type}, Coords: ({x},{y}) {width}x{height}, Timestamp: {timestamp}")

        # Load the screenshot
        screenshot_path = f"training_screenshot_{username}_{timestamp}.png"

        if not os.path.exists(screenshot_path):
            print(f"❌ Screenshot file not found: {screenshot_path}")
            print(f"Available files in directory: {os.listdir('.')}")
            return jsonify({"error": f"Screenshot not found: {screenshot_path}"}), 400

        print(f"✅ Found screenshot file: {screenshot_path}")
        image = cv2.imread(screenshot_path)

        if image is None:
            print(f"❌ Failed to load image from {screenshot_path}")
            return jsonify({"error": "Failed to load screenshot image"}), 400

        print(f"✅ Loaded image: {image.shape}")

        # Validate crop coordinates
        img_height, img_width = image.shape[:2]
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > img_width or y + height > img_height:
            print(f"❌ Invalid crop coordinates: ({x},{y}) {width}x{height} for image {img_width}x{img_height}")
            return jsonify({"error": "Invalid crop coordinates"}), 400

        print(f"✅ Valid crop coordinates: ({x},{y}) {width}x{height}")

        if template_type == 'head':
            success = training_system.save_head_template(image, x, y, width, height)
            if success:
                # Reload templates
                training_system.load_templates()
                template_count = len(training_system.get_head_templates())
                print(f"✅ Head template saved successfully. Total templates: {template_count}")
                return jsonify({"success": True, "message": f"Head template saved successfully. Total templates: {template_count}"})
            else:
                print("❌ Failed to save head template - training_system.save_head_template returned False")
                return jsonify({"error": "Failed to save head template"}), 400

        print(f"❌ Unknown template type: {template_type}")
        return jsonify({"error": "Unknown template type"}), 400

    except Exception as e:
        print(f"❌ Unexpected error in save_template: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

@app.route('/stop-fetch', methods=['POST'])
def stop_fetch():
    """Stop the current stream fetching process"""
    global fetch_status
    if not fetch_status["is_fetching"]:
        return jsonify({"error": "No scan currently running"}), 400

    fetch_status["stop_requested"] = True
    fetch_status["progress"] = "Stopping scan..."
    print("Stop signal sent to scanning process")
    return jsonify({"message": "Stop signal sent - scan will halt soon"})

@app.route('/batch-training', methods=['POST'])
def batch_training():
    """Start batch training session with multiple streams"""
    try:
        data = request.get_json()
        num_streams = min(int(data.get('num_streams', 10)), 20)  # Max 20 streams
        focus = data.get('focus', 'both')  # 'squads', 'kills', or 'both'

        # Get training streams (similar to regular scanning but prioritize variety)
        token = get_twitch_token()
        if not token:
            return jsonify({"error": "Failed to get Twitch token"}), 500

        streams = get_apex_streams(token, limit=num_streams)

        # Filter and prioritize streams for training
        training_streams = []
        for stream in streams[:num_streams]:
            training_streams.append({
                "username": stream["user_login"],
                "viewers": stream["viewer_count"],
                "url": f"https://www.twitch.tv/{stream['user_login']}"
            })

        return jsonify({
            "success": True,
            "training_streams": training_streams,
            "total_streams": len(training_streams),
            "focus": focus
        })

    except Exception as e:
        print(f"Batch training error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development, optionally run initial scan
    if os.environ.get('RUN_INITIAL_SCAN', 'false').lower() == 'true':
        fetch_and_update_streams()

    # Get port from environment (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() == 'true')
