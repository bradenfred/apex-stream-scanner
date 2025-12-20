# ------------------------------------------------------------------
#  STANDALONE OCR TUNER SCRIPT – WITH BATCH SUPPORT
# ------------------------------------------------------------------
#  Run this in a separate VS Code window to test OCR on saved images.
#  It loads one or more image files, applies preprocessing, runs OCR, and prints results.
#  Usage:
#    1. Set IMAGE_PATHS to a list of your crop_*.jpg or processed_*.jpg files (or a single string for one file).
#    2. Tweak tuning settings below (all optional preprocesses are off by default – enable them as needed).
#    3. Run: python this_script.py (in your .venv if using one)
#    4. Check console for raw text / detections + saved processed_{filename}.jpg for each image

import cv2
import pytesseract
import re
import os  # Added for batch file handling

# --------------------------------------------------------------
#  OCR TUNING SETTINGS – EDIT THESE VALUES ONLY
# --------------------------------------------------------------
# Preprocessing flags are all False by default – set to True to enable each step.
# Order in code: grayscale (always on) > noise filter > inversion > binarization > erosion > dilation > upscale.

# 1. APPLY NOISE FILTER? (after grayscale, to reduce speckles/artifacts)
#    True  → Enable noise reduction.
#    False → Skip (default).
APPLY_NOISE_FILTER = True

# 2. NOISE FILTER TYPE (only if APPLY_NOISE_FILTER = True)
#    "gaussian" → Smooths overall noise (good for general blur)
#    "median"   → Removes salt/pepper noise while preserving edges (good for text)
NOISE_FILTER_TYPE = "median"  # "gaussian" or "median"

# 3. NOISE FILTER KERNEL SIZE (odd number for median; tuple for gaussian, e.g., (5,5))
#    Larger → more smoothing (but may blur text too much)
#    Smaller → less smoothing
NOISE_KERNEL_SIZE = 1  # For median: int (odd); for gaussian: (int, int) like (3,3)

# 4. INVERT IMAGE? (after noise filter, for white text on dark bg)
#    True  → Flip colors (white → black, dark → white).
#    False → Skip (default).
INVERT_IMAGE = True

# 5. APPLY BINARISATION? (after inversion, to make black/white only)
#    True  → Enable binarization.
#    False → Skip (default – uses grayscale directly).
BINARIZE_METHOD = 'simple' # Set to "simple" or "adaptive" to enable; False to skip

# 6. SIMPLE THRESHOLD VALUE (0-255) – only if BINARIZE_METHOD = "simple"
#    Lower → more pixels become white (use when text is bright)
#    Higher → more pixels become black (use when text is dark)
SIMPLE_THRESHOLD = 80

# 7. ADAPTIVE THRESHOLD SETTINGS – only if BINARIZE_METHOD = "adaptive"
ADAPTIVE_METHOD = cv2.ADAPTIVE_THRESH_GAUSSIAN_C   # or ADAPTIVE_THRESH_MEAN_C
ADAPTIVE_BLOCK_SIZE = 11      # odd number, size of pixel neighbourhood
ADAPTIVE_C = 2                # constant subtracted from the mean

# 8. APPLY EROSION? (after binarization, to remove small noise or thin text)
#    True  → Enable erosion (shrinks white areas).
#    False → Skip (default).
APPLY_EROSION = True

# 9. EROSION KERNEL SIZE (tuple, e.g., (2,2)) and ITERATIONS
#    Larger kernel/iterations → more aggressive (may erase thin text)
EROSION_KERNEL_SIZE = (2,1)  # Shape of the structuring element
EROSION_ITERATIONS = 1        # Number of times to apply

# 10. APPLY DILATION? (after erosion, to thicken text)
#     True  → Enable dilation (expands white areas).
#     False → Skip (default).
APPLY_DILATION = True

# 11. DILATION KERNEL SIZE (tuple, e.g., (2,2)) and ITERATIONS
#     Larger kernel/iterations → more aggressive (may merge letters)
DILATION_KERNEL_SIZE = (2, 2)  # Shape of the structuring element
DILATION_ITERATIONS = 1        # Number of times to apply

# 12. UPSCALE FACTOR (always applied last – set to 1 to disable)
#     3–5 works well for tiny HUD text. Higher = more pixels → slower.
UPSCALE_FACTOR = 4  # Set to 1 for no upscale

# 13. OCR LANGUAGES (Tesseract traineddata files)
#     Add any language you need, e.g. 'eng+kor+jpn+fra'
OCR_LANGUAGES = 'eng+kor+jpn'

# 14. NUMBER OF OCR RETRIES (different PSM/OEM combos)
OCR_RETRIES = 4

# 15. TESSERACT PATH (if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 16. IMAGE PATHS TO TEST (now supports batch!)
#     Set this to a list of files, e.g., ['raw_crop_1.jpg', 'raw_crop_2.jpg']
#     Or a single string for one file.
#     Or a folder path (e.g., 'C:/path/to/folder/') to process all .jpg in it.
IMAGE_PATHS = 'images/processed_briannaelin.jpg'  # ← CHANGE THIS! (str, list, or folder path)

# --------------------------------------------------------------

def preprocess_for_ocr(image_bgr):
    """
    Applies grayscale + optional tuning steps to prepare image for OCR.
    Returns the final processed image (binary if binarized, else grayscale).
    Steps are in optimal order for easiest results: grayscale > noise > invert > binarize > erode > dilate > upscale.
    """
    # 1. Grayscale (always applied)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Noise filter (optional)
    if APPLY_NOISE_FILTER:
        if NOISE_FILTER_TYPE == "gaussian":
            gray = cv2.GaussianBlur(gray, NOISE_KERNEL_SIZE, 0)  # Expect tuple like (3,3)
        elif NOISE_FILTER_TYPE == "median":
            gray = cv2.medianBlur(gray, NOISE_KERNEL_SIZE)  # Expect int like 3

    # 3. Inversion (optional)
    if INVERT_IMAGE:
        gray = cv2.bitwise_not(gray)

    # 4. Binarisation (optional)
    if BINARIZE_METHOD:
        if BINARIZE_METHOD == "simple":
            _, binary = cv2.threshold(gray, SIMPLE_THRESHOLD, 255, cv2.THRESH_BINARY)
        else:  # adaptive
            binary = cv2.adaptiveThreshold(
                gray, 255, ADAPTIVE_METHOD, cv2.THRESH_BINARY, ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
            )
        processed = binary  # Use binary for further steps
    else:
        processed = gray  # Skip binarization – use grayscale

    # 5. Erosion (optional – after binarization or grayscale)
    if APPLY_EROSION:
        erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, EROSION_KERNEL_SIZE)
        processed = cv2.erode(processed, erosion_kernel, iterations=EROSION_ITERATIONS)

    # 6. Dilation (optional – after erosion)
    if APPLY_DILATION:
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, DILATION_KERNEL_SIZE)
        processed = cv2.dilate(processed, dilation_kernel, iterations=DILATION_ITERATIONS)

    # 7. Upscale (always applied unless factor=1)
    processed = cv2.resize(processed, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR, interpolation=cv2.INTER_LINEAR)

    return processed


def run_ocr_on_image(image_path):
    print(f"\n=== Testing OCR on: {image_path} ===\n")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("ERROR: Could not load image – check path!")
        return

    # ---- ROI (same as app – crop if needed, but assuming input is already cropped ----
    # If your test image isn't cropped, uncomment/adjust:
    # h, w, _ = frame.shape
    # crop_x = int(w * 0.8)
    # crop_y_start = int(h * 0.01)
    # crop_y_end = int(h * 0.08)
    # cropped = frame[crop_y_start:crop_y_end, crop_x:w]
    # ... then use cropped instead of frame below

    # Preprocess
    ocr_img = preprocess_for_ocr(frame)

    # Save processed for inspection
    filename = os.path.basename(image_path)
    debug_path = f'processed_{filename}'
    cv2.imwrite(debug_path, ocr_img)
    print(f"Saved processed image: {debug_path}")

    squads = None
    for attempt in range(OCR_RETRIES):
        print(f"OCR attempt {attempt + 1}/{OCR_RETRIES}")

        # Different configs per retry
        if attempt == 0:
            cfg = '--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789SQUADsLEFTleft'
        elif attempt == 1:
            cfg = '--psm 6 --oem 1 -c tessedit_char_whitelist=0123456789SQUADsLEFTleft'
        else:
            cfg = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789SQUADsLEFTleft'

        text = pytesseract.image_to_string(ocr_img, lang=OCR_LANGUAGES, config=cfg)
        print(f"  Raw text: '{text.strip()}'")

        m = re.search(r'(\d+)\s*squads?\s*(left|remaining)?', text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if val <= 30:
                squads = val
                print(f"  → Squads detected: {val}")
                break
            print(f"  → Invalid value {val} (too high)")

    if squads is None:
        print("  → No valid squad count detected")
    else:
        print(f"  → Final squads: {squads}")


if __name__ == '__main__':
    # Handle batch: if IMAGE_PATHS is str → list it; if folder → load all .jpg
    if isinstance(IMAGE_PATHS, str):
        if os.path.isdir(IMAGE_PATHS):
            IMAGE_PATHS = [os.path.join(IMAGE_PATHS, f) for f in os.listdir(IMAGE_PATHS) if f.lower().endswith('.jpg')]
            print(f"Batch mode: Found {len(IMAGE_PATHS)} JPG files in folder")
        else:
            IMAGE_PATHS = [IMAGE_PATHS]  # Single file as list

    for path in IMAGE_PATHS:
        run_ocr_on_image(path)