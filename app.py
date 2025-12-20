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
            variant_idx = int(variant_name.split('_v')[1])
            if variant_idx <= len(PREPROCESS_VARIANTS):
                ordered_variants.append(PREPROCESS_VARIANTS[variant_idx - 1])

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
    while len(all_streams) < limit and pages < 8:  # Increased to 8 pages for better coverage
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
    """Extract squads remaining from OCR text with flexible matching"""
    text_lower = text.lower()
    numbers = re.findall(r'\d+', text_lower)

    squads = []
    # More flexible patterns for squads
    squad_patterns = [
        r'(\d+)\s*squad',  # "3 squad"
        r'squad\s*(\d+)',  # "squad 3"
        r'(\d+)\s*left',   # "3 left"
        r'left\s*(\d+)',   # "left 3"
        r'(\d+)\s*remaining', # "3 remaining"
        r'remaining\s*(\d+)', # "remaining 3"
    ]

    for pattern in squad_patterns:
        matches = re.findall(pattern, text_lower)
        for match in matches:
            num_int = int(match)
            if 1 <= num_int <= 20:
                squads.append(num_int)

    # Also check for "squads left" style text
    if 'squad' in text_lower and ('left' in text_lower or 'remaining' in text_lower):
        # Extract any numbers in context
        for num in numbers:
            num_int = int(num)
            if 1 <= num_int <= 20:
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
    print("Running adaptive OCR with learning on frame...")
    try:
        height, width, _ = frame.shape

        # Crop for HUD info (upper-right corner - contains both squads and kills)
        hud_crop_x_start = int(width * 0.75)  # Start from 75% across screen
        hud_crop_y_start = int(height * 0.01)  # Start from 1% down
        hud_crop_y_end = int(height * 0.12)    # End at 12% down
        hud_crop_x_end = width                     # Go to right edge

        hud_cropped = frame[hud_crop_y_start:hud_crop_y_end, hud_crop_x_start:hud_crop_x_end]

        # Use adaptive preprocessing order based on learning
        preprocessing_order = ocr_learner.adaptive_preprocessing_order()

        # Process the entire HUD region for both squads and kills
        all_squads = []
        all_kills = []
        variant_results = []

        for idx, (prep_func, lang) in enumerate(preprocessing_order, 1):
            binary = prep_func(hud_cropped)

            # Run OCR and get confidence data
            try:
                # Get detailed OCR data with confidence
                ocr_data = pytesseract.image_to_data(binary, config='--psm 6', output_type=pytesseract.Output.DICT)
                text = ' '.join([word for word in ocr_data['text'] if word.strip()])
                confidences = ocr_data['conf']

                # Calculate average confidence (excluding -1 values)
                valid_confidences = [c for c in confidences if c >= 0]
                avg_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0

                print(f"Raw OCR text Tesseract (v{idx}): '{text.strip()}' (conf: {avg_confidence:.1f}%)")

                # Extract both squads and kills from the same OCR text
                squads_from_text = extract_squads_from_text(text)
                kills_from_text = extract_kills_from_text(text)

                # Also try visual kill detection using saved templates
                kills_from_visual = None
                if training_system.get_head_templates():
                    # Try visual detection with each saved head template
                    for head_template in training_system.get_head_templates():
                        visual_kills = detect_kills_visual(hud_cropped, head_template)
                        if visual_kills is not None:
                            kills_from_visual = visual_kills
                            print(f"Visual detection found {visual_kills} kills using template")
                            break

                # Combine text and visual kill detections
                all_kills_from_variant = kills_from_text[:]
                if kills_from_visual is not None:
                    all_kills_from_variant.append(kills_from_visual)

                # Record this detection for learning
                successful = bool(squads_from_text or all_kills_from_variant)
                ocr_learner.record_detection(
                    username=username,
                    preprocessing_variant=idx,
                    ocr_text=text,
                    confidence=avg_confidence / 100.0,  # Convert to 0-1 scale
                    squads_found=squads_from_text,
                    kills_found=all_kills_from_variant
                )

                variant_results.append({
                    'variant': idx,
                    'text': text,
                    'confidence': avg_confidence,
                    'squads': squads_from_text,
                    'kills': all_kills_from_variant,
                    'successful': successful
                })

                if squads_from_text:
                    print(f"Tesseract v{idx} detected squads: {squads_from_text}")
                    all_squads.extend(squads_from_text)

                if all_kills_from_variant:
                    print(f"Tesseract v{idx} detected kills: {all_kills_from_variant}")
                    all_kills.extend(all_kills_from_variant)

                # Early exit if we found good results with high confidence
                if successful and avg_confidence > 70:
                    print(f"High-confidence detection found, stopping early")
                    break

            except Exception as e:
                print(f"Tesseract error on variant {idx}: {e}")

        # Ensemble voting: if multiple variants agree on the same numbers, be more confident
        final_squads = None
        final_kills = None

        if all_squads:
            # Use majority voting for squads
            squad_counts = {}
            for squad in all_squads:
                squad_counts[squad] = squad_counts.get(squad, 0) + 1

            # Find most common squad count, but only if it appears multiple times
            max_count = max(squad_counts.values())
            if max_count >= 2:  # At least 2 variants agreed
                final_squads = [squad for squad, count in squad_counts.items() if count == max_count][0]
            else:
                # Fallback to filtering valid squads
                valid_squads = [s for s in all_squads if s <= 10]
                if valid_squads:
                    final_squads = min(valid_squads)

        if all_kills:
            # Use majority voting for kills
            kill_counts = {}
            for kill in all_kills:
                kill_counts[kill] = kill_counts.get(kill, 0) + 1

            # Find most common kill count, but only if it appears multiple times
            max_count = max(kill_counts.values())
            if max_count >= 2:  # At least 2 variants agreed
                final_kills = [kill for kill, count in kill_counts.items() if count == max_count][0]
            else:
                # Fallback to filtering valid kills
                valid_kills = [k for k in all_kills if k >= 7]  # Changed to >6 as user requested
                if valid_kills:
                    final_kills = max(valid_kills)

        # Print learning stats occasionally
        if len(ocr_learner.detection_history) % 50 == 0 and len(ocr_learner.detection_history) > 0:
            stats = ocr_learner.get_detection_stats()
            print(f"\n--- OCR Learning Stats ---")
            print(f"Total detections: {stats['total_detections']}")
            print(f"Success rate: {stats['success_rate']:.1%}")
            print(f"Best variants: {stats['best_variants']}")
            print("------------------------\n")

        if final_squads is not None:
            print(f"Final squads detected (after ensemble): {final_squads}")
        if final_kills is not None:
            print(f"Final kills detected (after ensemble): {final_kills}")

        if final_squads is None and final_kills is None:
            print("No valid squads or kills detected after ensemble voting.")
            return None

        return {"squads": final_squads, "kills": final_kills}
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

    with ThreadPoolExecutor(max_workers=12) as executor:  # Optimized for Railway free tier
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

@app.route('/capture-screenshot/<username>')
def capture_screenshot(username):
    """Capture a screenshot from a specific stream for training"""
    try:
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

        # Load the screenshot
        timestamp = data.get('timestamp')
        screenshot_path = f"training_screenshot_{username}_{timestamp}.png"

        if not os.path.exists(screenshot_path):
            return jsonify({"error": "Screenshot not found"}), 400

        image = cv2.imread(screenshot_path)

        if template_type == 'head':
            success = training_system.save_head_template(image, x, y, width, height)
            if success:
                # Reload templates
                training_system.load_templates()
                return jsonify({"success": True, "message": "Head template saved successfully"})
            else:
                return jsonify({"error": "Failed to save head template"}), 400

        return jsonify({"error": "Unknown template type"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # For local development, optionally run initial scan
    if os.environ.get('RUN_INITIAL_SCAN', 'false').lower() == 'true':
        fetch_and_update_streams()

    # Get port from environment (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'false').lower() == 'true')
