"""
Configuration file for Grocery Product Detection System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for Grocery Product Detection System"""

    # API Keys (set as environment variables or update here)
    # OPTION 1: Set environment variables (recommended for security):
    #   Windows: set ROBOFLOW_API_KEY=your_key_here
    #   PowerShell: $env:ROBOFLOW_API_KEY="your_key_here"
    #
    # OPTION 2: Replace the empty string "" below with your actual API key
    OPEN_FOOD_FACTS_API_URL = "https://world.openfoodfacts.org/api/v0/product"
    # <-- ADD YOUR ROBOFLOW API KEY HERE (replace "")
    ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
    ROBOFLOW_WORKSPACE = "new-workspace-wfzw3"
    ROBOFLOW_PROJECT = "grocery-dataset-q9fj2"
    ROBOFLOW_VERSION = 1

    # LLM Configuration
    LLM_PROVIDER = "gemini"  # Options: "gemini", "openai", "anthropic"
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

    # Model Paths
    MODEL_DIR = "models"
    DATASET_DIR = "datasets"
    MODEL_PATH = os.path.join(MODEL_DIR, "grocery_detector.pt")

    # Detection Thresholds
    BARCODE_CONFIDENCE_THRESHOLD = 0.5
    OBJECT_DETECTION_CONFIDENCE_THRESHOLD = 0.6  # Below 60%, use LLM fallback
    # Use LLM if confidence below 60% or product out of dataset
    LLM_FALLBACK_THRESHOLD = 0.6

    # Audio Settings
    AUDIO_ENABLED = True
    AUDIO_RATE = 150  # Speech rate (words per minute)
    AUDIO_VOICE_INDEX = 0  # Voice index for TTS

    # Camera Settings
    CAMERA_INDEX = 0  # 0 = default laptop webcam, change to 1 if external camera
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480

    # Output Settings
    VERBOSE = True


# For backward compatibility, also export as module-level variables
OPEN_FOOD_FACTS_API_URL = Config.OPEN_FOOD_FACTS_API_URL
ROBOFLOW_API_KEY = Config.ROBOFLOW_API_KEY
ROBOFLOW_WORKSPACE = Config.ROBOFLOW_WORKSPACE
ROBOFLOW_PROJECT = Config.ROBOFLOW_PROJECT
ROBOFLOW_VERSION = Config.ROBOFLOW_VERSION
LLM_PROVIDER = Config.LLM_PROVIDER
GEMINI_API_KEY = Config.GEMINI_API_KEY
OPENAI_API_KEY = Config.OPENAI_API_KEY
ANTHROPIC_API_KEY = Config.ANTHROPIC_API_KEY
MODEL_DIR = Config.MODEL_DIR
DATASET_DIR = Config.DATASET_DIR
MODEL_PATH = Config.MODEL_PATH
BARCODE_CONFIDENCE_THRESHOLD = Config.BARCODE_CONFIDENCE_THRESHOLD
OBJECT_DETECTION_CONFIDENCE_THRESHOLD = Config.OBJECT_DETECTION_CONFIDENCE_THRESHOLD
LLM_FALLBACK_THRESHOLD = Config.LLM_FALLBACK_THRESHOLD
AUDIO_ENABLED = Config.AUDIO_ENABLED
AUDIO_RATE = Config.AUDIO_RATE
AUDIO_VOICE_INDEX = Config.AUDIO_VOICE_INDEX
CAMERA_INDEX = Config.CAMERA_INDEX
CAMERA_WIDTH = Config.CAMERA_WIDTH
CAMERA_HEIGHT = Config.CAMERA_HEIGHT
VERBOSE = Config.VERBOSE
