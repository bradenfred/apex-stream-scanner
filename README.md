# Apex Legends Stream Scanner

A real-time web application that scans Twitch streams for Apex Legends matches nearing endgame or with high kill counts.

## 🚀 Quick Start

### 1. Double-click `start_scanner.bat`
This will launch the Flask server automatically.

### 2. Open your browser to: http://127.0.0.1:5000

### 3. Click "Scan Streams" to find matches!

## 📋 Setup (One-time only)

### Get Twitch API Credentials
1. Visit https://dev.twitch.tv/
2. Log in with your Twitch account
3. Go to "Your Console" → "Applications"
4. Click "Create Application"
5. Name: `Apex Stream Scanner`
6. OAuth Redirect: `http://localhost`
7. Copy Client ID and Client Secret

### Set Environment Variables
Create a file called `set_credentials.bat` and add:

```batch
set TWITCH_CLIENT_ID=your_client_id_here
set TWITCH_CLIENT_SECRET=your_client_secret_here
```

Run this batch file before starting the scanner.

## 🎯 Features

- **Real-time scanning** of live Apex streams
- **OCR detection** of squads remaining and kill counts
- **Smart filtering** for endgame (≤7 squads) and high-kill (>6 kills) matches
- **Auto-refresh** every 5 minutes
- **Responsive design** works on all devices
- **Keyboard shortcuts**: Ctrl+R (scan), Ctrl+1/2/3 (filters)

## 🛠️ How It Works

1. **Fetches live streams** from Twitch API
2. **Captures screenshots** of each stream
3. **Runs OCR** on HUD elements to detect:
   - Squads left (top-right corner)
   - Kill count (top-center area)
4. **Filters and sorts** results by priority
5. **Displays direct links** to watch qualifying matches

## 🎓 Learning Outcomes

This project demonstrates modern web development following industry best practices:

- **HTML**: Semantic structure with accessibility
- **CSS**: Modern styling with responsive design
- **JavaScript**: AJAX, state management, DOM manipulation
- **APIs**: RESTful endpoints with JSON responses
- **Python**: Backend processing with OCR and computer vision

## 🐛 Troubleshooting

- **"Twitch credentials not set"**: Run `set_credentials.bat` first
- **"HTTP 400" error**: Check your Twitch API credentials
- **No streams found**: Make sure Apex Legends is live on Twitch
- **OCR not working**: Ensure Tesseract OCR is installed

## 🚀 Cloud Deployment (Run 24/7 Online)

### Railway Deployment (Free & Recommended)

**Step-by-step guide:**

1. **Create Account**: Go to https://railway.app/ and sign up
2. **Connect GitHub**: Click "Start a New Project" → "Deploy from GitHub repo"
3. **Select Repository**: Choose your Apex Stream Scanner repository
4. **Auto-Deploy**: Railway will detect Python and install dependencies
5. **Set Environment Variables**: In Railway dashboard → Variables:
   ```
   TWITCH_CLIENT_ID = your_client_id_here
   TWITCH_CLIENT_SECRET = your_client_secret_here
   DEBUG = false
   ```
6. **Deploy**: Click "Deploy" - Railway gives you a live URL instantly!

**Your app will be available 24/7 at:** `https://your-project-name.up.railway.app`

**Cost:** Free for basic usage, scales automatically when needed.

### Option 2: Render (Also Free)

1. **Create Render Account**: https://render.com/
2. **New Web Service** → **Connect GitHub**
3. **Configure**:
   - Runtime: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
4. **Set Environment Variables** (same as above)
5. **Deploy**

### Option 3: Heroku

1. **Install Heroku CLI**
2. **Login**: `heroku login`
3. **Create App**: `heroku create your-app-name`
4. **Set Config**: `heroku config:set TWITCH_CLIENT_ID=your_id TWITCH_CLIENT_SECRET=your_secret`
5. **Deploy**: `git push heroku main`

## 📚 Requirements

- Python 3.7+
- Tesseract OCR (installed on deployment platform)
- Twitch API credentials
- Modern web browser

## 🤝 Contributing

Built following CTO learning roadmap:
1. HTML (structure)
2. CSS (styling)
3. JavaScript (behavior)
4. APIs (communication)
5. Python (backend processing)
