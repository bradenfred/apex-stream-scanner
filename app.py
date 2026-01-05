"""
Apex Legends Stream Scanner - AI Vision Agent Version
Uses Google Gemini Flash to analyze stream screenshots and extract game stats.
MEMORY OPTIMIZED for Railway deployment.
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
print("💾 Memory optimizations enabled")

# === MEMORY OPTIMIZATION SETTINGS ===
MAX_STREAMS_TO_SCAN = 50  # Reduced from 100
IMAGE_QUALITY = 65  # Reduced from 85
DELAY_BETWEEN_STREAMS = 0.5  # Seconds between each stream
MAX_IMAGE_WIDTH = 640  # Downscale images for AI analysis

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

# Background refresh state
refresh_status = {
    "is_refreshing": False,
    "last_refresh": "",
    "streams_checked": 0,
    "streams_removed": 0
}

# Processing details - MINIMAL storage to save memory
processing_details = {
    "current_stream": None,
    "status": "idle",
    "screenshot": None,  # Will be cleared after each stream
    "ai_response": "",
    "timeline": []
}


def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()


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
    
    try:
        response = requests.post(url, params=params, timeout=10)
        if response.status_code != 200:
            print(f"Token error: {response.status_code}")
            return None
        
        token = response.json().get("access_token")
        print(f"✅ Twitch token obtained")
        return token
    except Exception as e:
        print(f"Token error: {e}")
        return None


def get_apex_streams(token, limit=MAX_STREAMS_TO_SCAN):
    """
    Fetch live Apex Legends streams from Twitch with FAIR SELECTION.
    Collects a large pool of streams and selects randomly for equal probability.
    """
    if not token:
        return []

    url = "https://api.twitch.tv/helix/streams"
    headers = {
        "Authorization": f"Bearer {token}",
        "Client-Id": CLIENT_ID
    }

    # Step 1: Collect a large pool of ALL Apex streams (no viewer bias)
    all_streams = []
    cursor = None
    pages = 0
    max_pages = 8  # Reduced to 800 streams to prevent rate limiting

    print(f"📊 Collecting large pool of Apex streams (up to {max_pages * 100} streams)...")

    while pages < max_pages:
        params = {
            "game_id": "511224",  # Apex Legends
            "type": "live",
            "first": 100  # Always fetch 100 per page
        }
        if cursor:
            params["after"] = cursor

        try:
            response = requests.get(url, headers=headers, params=params, timeout=15)

            if response.status_code == 429:
                print(f"⚠️ Rate limited! Waiting before retry...")
                time.sleep(5)
                continue
            elif response.status_code != 200:
                print(f"API error: {response.status_code} - {response.text}")
                break

            data = response.json().get("data", [])
            if not data:
                print(f"   No more streams after page {pages + 1}")
                break

            streams = [{"user_login": s["user_login"], "viewer_count": s["viewer_count"]} for s in data]
            all_streams.extend(streams)

            cursor = response.json().get("pagination", {}).get("cursor")
            pages += 1

            print(f"   Page {pages}: Got {len(data)} streams (total pool: {len(all_streams)})")

            # Small delay between API calls to prevent rate limiting
            time.sleep(0.5)

            if not cursor:
                break

        except requests.exceptions.Timeout:
            print(f"⏰ Timeout on page {pages + 1}, stopping collection")
            break
        except Exception as e:
            print(f"API error on page {pages + 1}: {e}")
            break

    if len(all_streams) == 0:
        print("❌ No Apex streams found")
        return []

    # Step 2: Fair random selection from the complete pool
    # This gives equal probability to all streams regardless of viewer count
    if len(all_streams) <= limit:
        selected_streams = all_streams
        print(f"📊 Using all {len(all_streams)} streams (smaller than limit)")
    else:
        selected_streams = random.sample(all_streams, limit)
        print(f"🎲 Fair random selection: picked {limit} streams from {len(all_streams)} total pool")

    # Show viewer count distribution for transparency
    viewer_counts = [s["viewer_count"] for s in selected_streams]
    avg_viewers = sum(viewer_counts) / len(viewer_counts)
    min_viewers = min(viewer_counts)
    max_viewers = max(viewer_counts)

    print(f"📊 Selected streams stats: {min_viewers}-{max_viewers} viewers (avg: {avg_viewers:.0f})")

    return selected_streams


def downscale_image(frame, max_width=MAX_IMAGE_WIDTH):
    """Downscale image to save memory while preserving aspect ratio"""
    if frame is None:
        return None
    
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def capture_frame(username, timeout_seconds=8):
    """Capture a single frame from a Twitch stream - MEMORY OPTIMIZED"""
    print(f"📸 Capturing frame from {username}...")
    
    cap = None
    try:
        result = [None]
        error = [None]
        
        def capture_worker():
            nonlocal cap
            try:
                streams = streamlink.streams(f"https://www.twitch.tv/{username}")
                if not streams:
                    error[0] = "No streams available"
                    return
                
                # Use 480p quality to save memory (instead of 'best')
                quality = "480p" if "480p" in streams else "worst"
                stream_url = streams[quality].url
                
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
                
                ret, frame = cap.read()
                
                if cap:
                    cap.release()
                    cap = None
                
                if not ret:
                    error[0] = "Failed to read frame"
                    return
                
                result[0] = frame
                
            except Exception as e:
                error[0] = str(e)
                if cap:
                    cap.release()
        
        thread = threading.Thread(target=capture_worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_seconds)
        
        if thread.is_alive():
            print(f"⏰ Timeout capturing {username}")
            if cap:
                cap.release()
            return None
        
        if error[0]:
            print(f"❌ Capture error: {error[0]}")
            return None
        
        return result[0]
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        if cap:
            cap.release()
        return None


def analyze_stream_with_ai(frame, username):
    """Analyze a stream frame using the AI vision agent - MEMORY OPTIMIZED"""
    global processing_details
    
    try:
        # Downscale for AI analysis
        small_frame = downscale_image(frame)
        
        # Convert frame to bytes with lower quality
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        image_bytes = buffer.tobytes()
        
        # Clear buffer immediately
        del buffer
        del small_frame
        
        # Call the vision agent
        result = vision_agent.analyze_hud(image_bytes, username)
        
        # Clear image bytes
        del image_bytes
        
        # Minimal processing details update
        processing_details["ai_response"] = result.get("raw_response", "")[:200]  # Truncate
        
        return result
        
    except Exception as e:
        print(f"❌ AI analysis error: {e}")
        return {"squads": None, "players": None, "kills": None, "success": False}


def process_stream(stream):
    """Process a single stream with AI vision - MEMORY OPTIMIZED"""
    global processing_details

    username = stream["user_login"]
    stream_start = time.time()

    print(f"\n----- Processing: {username} (stream #{fetch_status['current_stream'] + 1}) -----")

    # Minimal monitoring update
    processing_details["current_stream"] = {
        "username": username,
        "viewers": stream["viewer_count"]
    }
    processing_details["status"] = "processing"
    processing_details["screenshot"] = None  # Clear old screenshot

    frame = None

    try:
        # Check if we should stop
        if fetch_status.get("stop_requested", False):
            print(f"🛑 Stop requested, aborting {username}")
            processing_details["status"] = "stopped"
            return None

        print(f"📸 Starting frame capture for {username}...")
        capture_start = time.time()

        # Capture frame
        frame = capture_frame(username)
        if frame is None:
            processing_details["status"] = "failed"
            print(f"❌ Frame capture failed for {username}")
            return None

        capture_time = time.time() - capture_start
        print(f"✅ Frame captured in {capture_time:.1f}s")

        # Analyze with AI (uses downscaled image internally)
        print(f"🤖 Starting AI analysis for {username}...")
        ai_start = time.time()

        result = analyze_stream_with_ai(frame, username)

        ai_time = time.time() - ai_start
        print(f"✅ AI analysis completed in {ai_time:.1f}s")

        # Clear frame immediately after analysis
        del frame
        frame = None
        cleanup_memory()

        if result.get("success"):
            squads = result.get("squads")
            players = result.get("players")
            kills = result.get("kills")
            assists = result.get("assists")
            damage = result.get("damage")
            is_ranked = result.get("is_ranked")

            print(f"✅ AI detected: Squads={squads}, Players={players}, Kills={kills}, Ranked={is_ranked}")

            # Check if AI found NO HUD elements (stream not showing gameplay)
            all_values_none = all([
                squads is None,
                players is None,
                kills is None,
                assists is None,
                damage is None
            ])

            if all_values_none:
                print(f"⏭️ No HUD detected - stream not showing gameplay (menu/loading/lobby)")
                processing_details["status"] = "no_hud"
                return None

            # Only qualify RANKED games (ignore casual/arena)
            if is_ranked != True:
                print(f"⏭️ Not ranked mode (ranked={is_ranked}) - skipping")
                processing_details["status"] = "not_ranked"
                return None

            # Check if this is a qualifying stream
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
                    "viewers": stream["viewer_count"],
                    "last_checked": time.strftime("%H:%M:%S")
                }

                # Add to results
                with streams_lock:
                    qualifying_streams.append(stream_data)
                    qualifying_streams.sort(
                        key=lambda x: (
                            x["squads"] if x["squads"] is not None else 999,
                            -(x["kills"] if x["kills"] is not None else 0)
                        )
                    )

                total_time = time.time() - stream_start
                print(f"🎯 QUALIFYING STREAM: Added to results (total time: {total_time:.1f}s)")
                return stream_data
            else:
                print(f"⏭️ Not qualifying (squads={squads}, kills={kills})")
                processing_details["status"] = "no_content"
                return None
        else:
            print(f"❌ AI analysis failed for {username}")
            processing_details["status"] = "failed"
            return None

    except Exception as e:
        print(f"💥 CRITICAL ERROR processing {username}: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        processing_details["status"] = "error"
        return None

    finally:
        # Ensure frame is cleaned up
        if frame is not None:
            del frame
        cleanup_memory()

        total_time = time.time() - stream_start
        print(f"🏁 Finished processing {username} in {total_time:.1f}s")


def fetch_and_update_streams():
    """Main function to fetch and process streams - MEMORY OPTIMIZED"""
    global qualifying_streams, last_updated, fetch_status

    scan_start_time = time.time()
    MAX_SCAN_TIME = 600  # 10 minutes max to prevent Railway timeouts

    print("\n===== Starting Stream Scan =====\n")

    fetch_status["progress"] = "Getting Twitch token..."
    token = get_twitch_token()

    if not token:
        fetch_status["is_fetching"] = False
        fetch_status["progress"] = "Failed to get Twitch token"
        return

    fetch_status["progress"] = "Fetching Apex streams..."
    streams = get_apex_streams(token, limit=MAX_STREAMS_TO_SCAN)

    fetch_status["total_streams"] = len(streams)

    # Clear old results
    with streams_lock:
        qualifying_streams.clear()

    cleanup_memory()

    completed = 0
    for stream in streams:
        # Check for timeout
        elapsed_time = time.time() - scan_start_time
        if elapsed_time > MAX_SCAN_TIME:
            print(f"⏰ Scan timeout reached ({MAX_SCAN_TIME}s), stopping...")
            fetch_status["progress"] = f"Timeout! Processed {completed}/{len(streams)} streams..."
            break

        if fetch_status["stop_requested"]:
            print("🛑 Stop requested")
            break

        stream_start = time.time()
        process_stream(stream)
        stream_time = time.time() - stream_start

        completed += 1
        fetch_status["current_stream"] = completed

        # Estimate remaining time
        if completed > 0:
            avg_time_per_stream = elapsed_time / completed
            remaining_streams = len(streams) - completed
            estimated_remaining = remaining_streams * avg_time_per_stream
            fetch_status["progress"] = f"Processed {completed}/{len(streams)} streams... (~{estimated_remaining:.0f}s left)"
        else:
            fetch_status["progress"] = f"Processed {completed}/{len(streams)} streams..."

        print(f"📊 Progress: {completed}/{len(streams)} streams ({elapsed_time:.0f}s elapsed)")

        # Memory cleanup between streams
        cleanup_memory()

        # Small delay to prevent overload
        time.sleep(DELAY_BETWEEN_STREAMS)

    total_scan_time = time.time() - scan_start_time

    with streams_lock:
        last_updated = time.strftime("%Y-%m-%d %H:%M:%S")

    fetch_status["is_fetching"] = False
    fetch_status["progress"] = f"Complete! Found {len(qualifying_streams)} qualifying streams in {total_scan_time:.0f}s."

    # Final cleanup
    cleanup_memory()

    print(f"\n===== Scan Complete: {len(qualifying_streams)} qualifying streams in {total_scan_time:.0f}s =====\n")


def refresh_existing_streams():
    """Re-check existing qualifying streams - MEMORY OPTIMIZED"""
    global qualifying_streams, refresh_status
    
    refresh_status["is_refreshing"] = True
    refresh_status["streams_checked"] = 0
    refresh_status["streams_removed"] = 0
    
    print("\n🔄 Refreshing existing streams...")
    
    with streams_lock:
        streams_to_check = list(qualifying_streams)
    
    updated_streams = []
    
    for stream_data in streams_to_check:
        username = stream_data["user"]
        refresh_status["streams_checked"] += 1
        
        print(f"🔍 Re-checking {username}...")
        
        frame = None
        try:
            frame = capture_frame(username, timeout_seconds=6)
            if frame is None:
                print(f"  ❌ {username} - offline, removing")
                refresh_status["streams_removed"] += 1
                continue
            
            # Analyze with AI (uses downscaled image internally)
            small_frame = downscale_image(frame)
            _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
            result = vision_agent.analyze_hud(buffer.tobytes(), username)
            
            del buffer
            del small_frame
            del frame
            frame = None
            
            if result.get("success"):
                squads = result.get("squads")
                kills = result.get("kills")
                damage = result.get("damage")
                is_ranked = result.get("is_ranked")

                # Only keep RANKED games during refresh
                if is_ranked != True:
                    print(f"  ⏭️ {username} - no longer ranked, removing")
                    refresh_status["streams_removed"] += 1
                    continue

                is_endgame = squads is not None and squads <= 10
                has_high_kills = kills is not None and kills >= 5
                has_high_damage = damage is not None and damage >= 500

                if is_endgame or has_high_kills or has_high_damage:
                    stream_data["squads"] = squads
                    stream_data["players"] = result.get("players")
                    stream_data["kills"] = kills
                    stream_data["assists"] = result.get("assists")
                    stream_data["damage"] = damage
                    stream_data["last_checked"] = time.strftime("%H:%M:%S")
                    updated_streams.append(stream_data)
                    print(f"  ✅ {username} - still qualifying")
                else:
                    print(f"  ⏭️ {username} - no longer qualifying")
                    refresh_status["streams_removed"] += 1
            else:
                updated_streams.append(stream_data)
                print(f"  ⚠️ {username} - AI failed, keeping")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            updated_streams.append(stream_data)
        finally:
            if frame is not None:
                del frame
            cleanup_memory()
    
    with streams_lock:
        qualifying_streams.clear()
        qualifying_streams.extend(updated_streams)
        qualifying_streams.sort(
            key=lambda x: (
                x["squads"] if x["squads"] is not None else 999,
                -(x["kills"] if x["kills"] is not None else 0)
            )
        )
    
    refresh_status["is_refreshing"] = False
    refresh_status["last_refresh"] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    cleanup_memory()
    
    print(f"🔄 Refresh complete: {len(updated_streams)} remaining, {refresh_status['streams_removed']} removed")


# Flask Routes

@app.before_request
def handle_options():
    """Handle CORS preflight requests"""
    if request.method == 'OPTIONS':
        response = make_response()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response

@app.after_request
def add_cors_headers(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

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
        # Only log status requests when scanning or when progress changes
        global last_logged_progress
        if 'last_logged_progress' not in globals():
            last_logged_progress = None

        current_progress = fetch_status["progress"]
        should_log = (
            fetch_status["is_fetching"] or  # Always log when scanning
            current_progress != last_logged_progress or  # Log when progress changes
            last_logged_progress is None  # Log first request
        )

        if should_log:
            print(f"📊 Status requested: {current_progress}")
            last_logged_progress = current_progress

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

    try:
        print(f"🔄 Fetch request received from {request.remote_addr}")
        print(f"   Headers: {dict(request.headers)}")
        print(f"   Data: {request.get_data(as_text=True)}")

        # Check if already scanning
        if fetch_status["is_fetching"]:
            print("❌ Rejecting: Already scanning")
            return jsonify({"error": "Already scanning"}), 400

        # Check Twitch credentials
        if not CLIENT_ID or not CLIENT_SECRET:
            print("❌ Rejecting: Twitch credentials not configured")
            print(f"   CLIENT_ID: {'SET' if CLIENT_ID else 'MISSING'}")
            print(f"   CLIENT_SECRET: {'SET' if CLIENT_SECRET else 'MISSING'}")
            return jsonify({"error": "Twitch credentials not configured"}), 400

        # Check Gemini API
        if not vision_agent.is_available():
            print("❌ Rejecting: Vision agent not configured")
            return jsonify({"error": "Vision agent not configured. Set GEMINI_API_KEY."}), 400

        print("✅ All validations passed, starting scan...")

        # Reset fetch status
        fetch_status = {
            "is_fetching": True,
            "progress": "Starting...",
            "current_stream": 0,
            "total_streams": 0,
            "stop_requested": False
        }

        # Start the scanning thread
        try:
            thread = threading.Thread(target=fetch_and_update_streams, daemon=True)
            thread.start()
            print("✅ Scan thread started successfully")
        except Exception as e:
            print(f"❌ Failed to start scan thread: {e}")
            fetch_status["is_fetching"] = False
            return jsonify({"error": f"Failed to start scanning: {str(e)}"}), 500

        return jsonify({"message": "Stream scanning started"})

    except Exception as e:
        print(f"❌ Unexpected error in fetch endpoint: {e}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/stop-fetch', methods=['POST'])
def stop_fetch():
    global fetch_status
    
    if not fetch_status["is_fetching"]:
        return jsonify({"error": "No scan running"}), 400
    
    fetch_status["stop_requested"] = True
    return jsonify({"message": "Stop signal sent"})


@app.route('/refresh-streams', methods=['POST'])
def refresh_streams():
    global refresh_status
    
    if refresh_status["is_refreshing"]:
        return jsonify({"error": "Already refreshing"}), 400
    
    if not qualifying_streams:
        return jsonify({"error": "No streams to refresh"}), 400
    
    thread = threading.Thread(target=refresh_existing_streams, daemon=True)
    thread.start()
    
    return jsonify({
        "message": f"Refreshing {len(qualifying_streams)} streams...",
        "streams_count": len(qualifying_streams)
    })


@app.route('/refresh-status')
def get_refresh_status():
    return jsonify({
        "is_refreshing": refresh_status["is_refreshing"],
        "last_refresh": refresh_status["last_refresh"],
        "streams_checked": refresh_status["streams_checked"],
        "streams_removed": refresh_status["streams_removed"],
        "current_streams": len(qualifying_streams)
    })


@app.route('/test-ai/<username>')
def test_ai(username):
    """Test AI vision on a specific stream"""
    frame = None
    try:
        frame = capture_frame(username)
        if frame is None:
            return jsonify({"error": "Failed to capture frame"}), 400
        
        small_frame = downscale_image(frame)
        _, buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, IMAGE_QUALITY])
        
        result = vision_agent.analyze_hud(buffer.tobytes(), username)
        
        # Small thumbnail for response
        _, thumb = cv2.imencode('.jpg', downscale_image(frame, 320), [cv2.IMWRITE_JPEG_QUALITY, 50])
        img_base64 = base64.b64encode(thumb).decode('utf-8')
        
        del buffer
        del small_frame
        del frame
        frame = None
        cleanup_memory()
        
        return jsonify({
            "success": True,
            "username": username,
            "ai_result": result,
            "thumbnail": f"data:image/jpeg;base64,{img_base64}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if frame is not None:
            del frame
        cleanup_memory()


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "vision_agent": vision_agent.is_available(),
        "twitch_configured": bool(CLIENT_ID and CLIENT_SECRET),
        "memory_optimized": True,
        "max_streams": MAX_STREAMS_TO_SCAN,
        "timestamp": int(time.time()),
        "active_streams": len(qualifying_streams),
        "is_scanning": fetch_status["is_fetching"]
    })


@app.route('/ping')
def ping():
    """Simple ping endpoint for Railway health checks"""
    return jsonify({"pong": True, "timestamp": int(time.time())})


@app.route('/ocr-stats')
def ocr_stats():
    """OCR learning statistics endpoint"""
    # Return mock stats for now - in a real implementation this would track
    # OCR success rates, best preprocessing methods, etc.
    return jsonify({
        "ocr_learning_stats": {
            "total_detections": len(qualifying_streams) * 3,  # Rough estimate
            "success_rate": 0.85,  # 85% success rate
            "best_variants": [
                ["default", 0.88, 45],
                ["enhanced_contrast", 0.82, 38],
                ["noise_reduction", 0.79, 32]
            ]
        },
        "streams_analyzed": len(qualifying_streams),
        "last_updated": last_updated
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
