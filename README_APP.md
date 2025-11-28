# Emo-TTS Frontend & Backend Setup

## Prerequisites
1.  **FFmpeg**: Required for audio processing.
    ```bash
    brew install ffmpeg
    ```
2.  **Google Gemini API Key**:
    ```bash
    export GOOGLE_API_KEY='your_key_here'
    ```
3.  **Python & UV**: Ensure you have Python installed. We recommend using `uv` for package management.
    ```bash
    pip install uv
    ```

## Installation

### 1. Backend Setup
```bash
cd sentiment
uv pip install -r requirements.txt
```

### 2. Frontend Setup
```bash
cd frontend
npm install
```

## How to Run

### 1. Start the Backend (Python API)
Open a terminal and run:
```bash
cd sentiment
uv run uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
*This starts the Neuro-Adaptive Brain on http://localhost:8000*

### 2. Start the Frontend (React App)
Open a **new** terminal window and run:
```bash
cd frontend
npm run dev
```
*This starts the UI on http://localhost:5173 (or 5174 if 5173 is busy)*

## Usage
1.  Open the URL shown in the frontend terminal (e.g., `http://localhost:5173`) in your browser.
2.  Click the **Microphone Icon** to start recording.
3.  Speak into your microphone.
4.  Click the **Square Icon** to stop and send.
5.  Watch the "Amygdala State" update and read the AI's analysis and response.
