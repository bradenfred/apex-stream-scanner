"""
Vision Agent for Apex Legends Stream Scanner
Uses Google Gemini Flash to analyze stream screenshots and extract game stats.
"""

import os
import base64
import json
import re
import logging
from io import BytesIO
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Gemini API configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = "gemini-2.0-flash-exp"  # Fast and cheap

# Import Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Vision agent will not work.")


def configure_gemini():
    """Configure Gemini API with the API key."""
    if not GEMINI_AVAILABLE:
        logger.error("Gemini SDK not available")
        return False
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY environment variable not set")
        return False
    
    genai.configure(api_key=GEMINI_API_KEY)
    return True


def crop_top_right_quarter(image_data: bytes) -> bytes:
    """
    Crop the top-right quarter of an image.
    This is where Apex Legends displays squads, players, and kills.
    
    Args:
        image_data: Raw image bytes (JPEG/PNG)
    
    Returns:
        Cropped image as bytes
    """
    try:
        img = Image.open(BytesIO(image_data))
        width, height = img.size
        
        # Crop top-right quarter (right half, top half)
        left = width // 2
        top = 0
        right = width
        bottom = height // 2
        
        cropped = img.crop((left, top, right, bottom))
        
        # Convert back to bytes
        output = BytesIO()
        cropped.save(output, format='JPEG', quality=85)
        output.seek(0)
        return output.read()
    
    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        return image_data  # Return original if cropping fails


def analyze_hud(image_data: bytes, username: str = "unknown") -> dict:
    """
    Analyze an Apex Legends HUD screenshot using Gemini Flash.
    
    Args:
        image_data: Raw screenshot bytes
        username: Streamer username for logging
    
    Returns:
        Dictionary with extracted stats:
        {
            "squads": int or None,
            "players": int or None,
            "kills": int or None,
            "raw_response": str,
            "success": bool
        }
    """
    result = {
        "squads": None,
        "players": None,
        "kills": None,
        "assists": None,
        "damage": None,
        "is_ranked": None,  # True if ranked mode, False if casual, None if unknown
        "raw_response": "",
        "success": False
    }
    
    if not configure_gemini():
        result["raw_response"] = "Gemini API not configured"
        return result
    
    try:
        # Crop to top-right quarter where HUD is located
        cropped_data = crop_top_right_quarter(image_data)
        
        # Create the model
        model = genai.GenerativeModel(GEMINI_MODEL)
        
        # Create image part for the API
        image_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(cropped_data).decode('utf-8')
        }
        
        # Prompt optimized for Apex Legends HUD extraction with ranked mode detection
        prompt = """Analyze this Apex Legends game screenshot and extract the following information from the HUD.

Look for game mode indicators:
- RANKED MODE: Look for "RANKED" text, crown/star icons, tier names (Bronze, Silver, Gold, Platinum, Diamond, Master, Apex Predator), or RP (ranked points) display
- CASUAL MODE: Look for "CASUAL" text or arena mode indicators

The Apex HUD typically shows in the top area:
- "X SQUADS LEFT" text with a player count number next to it
- Below that: Kills (skull icon), Assists, Participation icons with numbers
- Damage number (usually to the right)

Extract these values:
1. SQUADS: The number before "SQUADS LEFT" (e.g., "9 SQUADS LEFT" = 9)
2. PLAYERS: The number next to the player silhouette icon (shows total players alive)
3. KILLS: The number next to the skull icon (usually 0-30+)
4. ASSISTS: The number of assists (optional)
5. DAMAGE: The damage number (usually 0-3000+)
6. IS_RANKED: true if this is ranked mode, false if casual/arena, null if unknown

Return ONLY a JSON object in this exact format, nothing else:
{"squads": NUMBER_OR_NULL, "players": NUMBER_OR_NULL, "kills": NUMBER_OR_NULL, "assists": NUMBER_OR_NULL, "damage": NUMBER_OR_NULL, "is_ranked": BOOLEAN_OR_NULL}

If you cannot find a value, use null. Only use integers for numeric values, booleans for is_ranked.
If this is not gameplay (loading screen, menu, lobby, etc.), return all nulls."""

        # Call Gemini API
        response = model.generate_content([prompt, image_part])
        
        raw_text = response.text.strip()
        result["raw_response"] = raw_text
        
        logger.info(f"[{username}] Gemini response: {raw_text}")
        
        # Parse the JSON response
        # Try to extract JSON from the response (handle markdown code blocks)
        json_match = re.search(r'\{[^}]+\}', raw_text)
        if json_match:
            json_str = json_match.group()
            parsed = json.loads(json_str)
            
            # Extract values
            if "squads" in parsed and parsed["squads"] is not None:
                result["squads"] = int(parsed["squads"])
            
            if "players" in parsed and parsed["players"] is not None:
                result["players"] = int(parsed["players"])
            
            if "kills" in parsed and parsed["kills"] is not None:
                result["kills"] = int(parsed["kills"])
            
            if "assists" in parsed and parsed["assists"] is not None:
                result["assists"] = int(parsed["assists"])
            
            if "damage" in parsed and parsed["damage"] is not None:
                result["damage"] = int(parsed["damage"])

            if "is_ranked" in parsed and parsed["is_ranked"] is not None:
                result["is_ranked"] = bool(parsed["is_ranked"])

            result["success"] = any([
                result["squads"] is not None,
                result["players"] is not None,
                result["kills"] is not None
            ])
        
        logger.info(f"[{username}] Extracted: squads={result['squads']}, players={result['players']}, kills={result['kills']}, assists={result['assists']}, damage={result['damage']}, is_ranked={result['is_ranked']}")
        
    except json.JSONDecodeError as e:
        logger.error(f"[{username}] Failed to parse Gemini JSON response: {e}")
        result["raw_response"] = f"JSON parse error: {e}"
    
    except Exception as e:
        logger.error(f"[{username}] Gemini API error: {e}")
        result["raw_response"] = f"API error: {e}"
    
    return result


def is_available() -> bool:
    """Check if the vision agent is properly configured."""
    return GEMINI_AVAILABLE and bool(GEMINI_API_KEY)


# Quick test function
if __name__ == "__main__":
    print(f"Gemini SDK available: {GEMINI_AVAILABLE}")
    print(f"API key configured: {bool(GEMINI_API_KEY)}")
    print(f"Vision agent ready: {is_available()}")
