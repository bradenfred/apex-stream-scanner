"""
Apex Legends Stream Scanner - AI Vision Agent Version
Uses Google Gemini Flash to analyze stream screenshots and extract game stats.
"""

import os
import re
import time
import random
import requests
import cv2
import streamlink
from flask import Flask, render_template, redirect, url_for, jsonify, request, make_response
import threading
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import json
import base64

# Import vision agent
import vision_agent

# Twitch API credentials
CLIENT_ID = os.environ.get('TWITCH_CLIENT_ID')
CLIENT_SECRET = os.environ.get('TWITCH_CLIENT_SECRET')

if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: Twitch API credentials not set!")
    print("Set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET environment variables.")

app = Flask(__name__)

print("🤖 Running with AI Vision Agent (Google Gemini Flash)")
print(f"   Vision agent available: {vision_agent.is_available()}")

# Global state
qualifying_streams = []
last_updated = ""
fetch_status = {
    "is_fetching": False, 
    "progress": "", 
    "current_stream": 0, 
    "total_streams": 0, 
    "stop_requested": False
}
streams_lock = threading.Lock()

# Processing details for monitoring
processing_details = {
    "current_stream": None,
    "status": "idle",
    "screenshot": None,
    "ai_response": "",
    "extracted_data": {
        "squads": None,
        "players": None,
        "kills": None
    },
    "timeline": [],
    "processing_log": []
}


def get_twitch_token():
    """Get OAuth token from Twitch API"""
    if not CLIENT_ID or not CLIENT_SECRET:
        print("ERROR: Twitch credentials not set")
        return None

    url = "https://id.twitch.tv/oauth2/token"
    params = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials"
    }
    
    response = requests.post(url, params=params)
    if response.status_code != 200:
        print(f"Token error: {response.status_code}")
        return None
    
    token = response.json().get("access_token")
    print(f"✅ Twitch token obtained")
    return token


def get_apex_streams(token, limit=100):
    """Fetch live Apex Legends streams from Twitch"""
    if not token:
        return []
    
    url = "https://api.twitch.tv/helix/streams"
    headers = {
        "Authorization": f"Bearer {token}",
        "Client-Id": CLIENT_ID
    }
    
    all_streams = []
    cursor = None
    pages = 0
    
    while len(all_streams) < limit and pages < 4:
        params = {
            "game_id": "511224",  # Apex Legends
            "type": "live",
            "first": 100
        }
        if cursor:
            params["after"] = cursor
        
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code != 200:
            print(f"API error: {response.status_code}")
            break
        
        data = response.json().get("data", [])
        if not data:
            break
        
        streams = [{"user_login": s["user_login"], "viewer_count": s["viewer_count"]} for s in data]
        all_streams.extend(streams)
        
        cursor = response.json().get("pagination", {}).get("cursor")
        pages += 1
        
        if not cursor:
            break
    
    # Sort by viewer count (prioritize smaller streams)
    all_streams.sort(key=lambda x: x["viewer_count"])
    random.shuffle(all_streams)
    
    print(f"📡 Found {len(all_streams)} Apex streams")
    return all_streams[:limit]


def capture_frame(username, timeout_seconds=10):
    """Capture a single frame from a Twitch stream"""
    print(f"📸 Capturing frame from {username}...")
    
    try:
        result = [None]
        error = [None]
        
        def capture_worker():
            try:
                streams = streamlink.streams(f"https://www.twitch.tv/{username}")
                if not streams:
                    error[0] = "No streams available"
                    return
                
                stream_url = streams["best"].url
                cap = cv2.VideoCapture(stream_url)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    error[0] = "Failed to read frame"
                    return
                
                result[0] = frame
                
            except Exception as e:
                error[0] = str(e)
        
        thread = threading.Thread(target=capture_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            print(f"⏰ Timeout capturing {username}")
            return None
        
        if error[0]:
            print(f"❌ Capture error: {error[0]}")
            return None
        
        return result[0]
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None


def analyze_stream_with_ai(frame, username):
    """Analyze a stream frame using the AI vision agent"""
    global processing_details
    
    try:
        # Convert frame to bytes for the vision agent
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = buffer.tobytes()
        
        # Call the vision agent
        result = vision_agent.analyze_hud(image_bytes, username)
        
        # Update processing details
        processing_details["ai_response"] = result.get("raw_response", "")
        processing_details["extracted_data"] = {
            "squads": result.get("squads"),
            "players": result.get("players"),
            "kills": result.get("kills")
        }
        
        return result
        
    except Exception as e:
        print(f"❌ AI analysis error: {e}")
        return {"squads": None, "players": None, "kills": None, "success": False}


def process_stream(stream):
    """Process a single stream with AI vision"""
    global processing_details
    
    username = stream["user_login"]
    print(f"\n----- Processing: {username} -----")
    
    # Update monitoring
    processing_details["current_stream"] = {
        "username": username,
        "viewers": stream["viewer_count"],
        "url": f"https://www.twitch.tv/{username}"
    }
    processing_details["status"] = "processing"
    processing_details["timeline"] = []
    start_time = time.time()
    
    def log(msg):
        elapsed = time.time() - start_time
        processing_details["timeline"].append(f"[{elapsed:.1f}s] {msg}")
        print(msg)
    
    log("Starting capture...")
    
    # Capture frame
    frame = capture_frame(username)
    if frame is None:
        log("❌ Frame capture failed")
        processing_details["status"] = "failed"
        return None
    
    log("✅ Frame captured")
    
    # Store screenshot for monitoring
    _, buffer = cv2.imencode('.png', frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    processing_details["screenshot"] = f"data:image/png;base64,{img_base64}"
    
    log("🤖 Analyzing with AI vision agent...")
    
    # Analyze with AI
    result = analyze_stream_with_ai(frame, username)
    
    if result.get("success"):
        squads = result.get("squads")
        players = result.get("players")
        kills = result.get("kills")
        assists = result.get("assists")
        damage = result.get("damage")
        
        log(f"✅ AI detected: Squads={squads}, Players={players}, Kills={kills}, Assists={assists}, Damage={damage}")
        
        # Check if this is a qualifying stream (endgame or high kills/damage)
        is_endgame = squads is not None and squads <= 10
        has_high_kills = kills is not None and kills >= 5
        has_high_damage = damage is not None and damage >= 500
        
        if is_endgame or has_high_kills or has_high_damage:
            processing_details["status"] = "completed"
            
            stream_data = {
                "user": username,
                "squads": squads,
                "players": players,
                "kills": kills,
                "assists": assists,
                "damage": damage,
                "url": f"https://www.twitch.tv/{username}",
                "viewers": stream["viewer_count"]
            }
            
            # Add to results immediately
            with streams_lock:
                qualifying_streams.append(stream_data)
                # Sort by squads (endgame first), then by kills
                qualifying_streams.sort(
                    key=lambda x: (
                        x["squads"] if x["squads"] is not None else 999,
                        -(x["kills"] if x["kills"] is not None else 0)
                    )
                )
            
            log(f"🎯 QUALIFYING STREAM: Added to results")
            return stream_data
        else:
            log(f"⏭️ Not qualifying (squads={squads}, kills={kills})")
    else:
        log(f"❌ AI analysis failed: {result.get('raw_response', 'Unknown error')}")
    
    processing_details["status"] = "no_content"
    return None


def fetch_and_update_streams():
    """Main function to fetch and process streams"""
    global qualifying_streams, last_updated, fetch_status
    
    print("\n===== Starting Stream Scan =====\n")
    
    fetch_status["progress"] = "Getting Twitch token..."
    token = get_twitch_token()
    
    if not token:
        fetch_status["is_fetching"] = False
        fetch_status["progress"] = "Failed to get Twitch token"
        return
    
    fetch_status["progress"] = "Fetching Apex streams..."
    streams = get_apex_streams(token)
    
    fetch_status["total_streams"] = len(streams)
    
    # Clear old results for fresh scan
    with streams_lock:
        qualifying_streams.clear()
    
    completed = 0
    for stream in streams:
        if fetch_status["stop_requested"]:
            print("🛑 Stop requested")
            break
        
        process_stream(stream)
        completed += 1
        fetch_status["current_stream"] = completed
        fetch_status["progress"] = f"Processed {completed}/{len(streams)} streams..."
    
    with streams_lock:
        last_updated = time.strftime("%Y-%m-%d %H:%M:%S")
    
    fetch_status["is_fetching"] = False
    fetch_status["progress"] = f"Complete! Found {len(qualifying_streams)} qualifying streams."
    print(f"\n===== Scan Complete: {len(qualifying_streams)} qualifying streams =====\n")


# Flask Routes

@app.route('/')
def home():
    response = make_response(render_template(
        'index.html', 
        streams=qualifying_streams, 
        last_updated=last_updated, 
        version=int(time.time())
    ))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


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
        return jsonify({"error": "Already scanning"}), 400
    
    if not CLIENT_ID or not CLIENT_SECRET:
        return jsonify({"error": "Twitch credentials not configured"}), 400
    
    if not vision_agent.is_available():
        return jsonify({"error": "Vision agent not configured. Set GEMINI_API_KEY."}), 400
    
    fetch_status = {
        "is_fetching": True, 
        "progress": "Starting...", 
        "current_stream": 0, 
        "total_streams": 0, 
        "stop_requested": False
    }
    
    thread = threading.Thread(target=fetch_and_update_streams)
    thread.start()
    
    return jsonify({"message": "Stream scanning started"})


@app.route('/stop-fetch', methods=['POST'])
def stop_fetch():
    global fetch_status
    
    if not fetch_status["is_fetching"]:
        return jsonify({"error": "No scan running"}), 400
    
    fetch_status["stop_requested"] = True
    return jsonify({"message": "Stop signal sent"})


@app.route('/processing-details')
def processing_details_endpoint():
    return jsonify(processing_details)


@app.route('/capture-screenshot/<username>')
def capture_screenshot(username):
    """Capture screenshot for training/testing"""
    try:
        frame = capture_frame(username)
        if frame is None:
            return jsonify({"error": "Failed to capture frame"}), 400
        
        timestamp = int(time.time())
        
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


@app.route('/test-ai/<username>')
def test_ai(username):
    """Test AI vision on a specific stream"""
    try:
        frame = capture_frame(username)
        if frame is None:
            return jsonify({"error": "Failed to capture frame"}), 400
        
        # Convert to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        image_bytes = buffer.tobytes()
        
        # Analyze with AI
        result = vision_agent.analyze_hud(image_bytes, username)
        
        # Include screenshot in response
        _, png_buffer = cv2.imencode('.png', frame)
        img_base64 = base64.b64encode(png_buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "username": username,
            "ai_result": result,
            "screenshot": f"data:image/png;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/batch-training', methods=['POST'])
def batch_training():
    """Get streams for batch training"""
    try:
        data = request.get_json()
        num_streams = min(int(data.get('num_streams', 10)), 20)
        
        token = get_twitch_token()
        if not token:
            return jsonify({"error": "Failed to get Twitch token"}), 500
        
        streams = get_apex_streams(token, limit=num_streams)
        
        training_streams = [{
            "username": s["user_login"],
            "viewers": s["viewer_count"],
            "url": f"https://www.twitch.tv/{s['user_login']}"
        } for s in streams]
        
        return jsonify({
            "success": True,
            "training_streams": training_streams,
            "total_streams": len(training_streams)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "vision_agent": vision_agent.is_available(),
        "twitch_configured": bool(CLIENT_ID and CLIENT_SECRET)
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
