"""
Audio Output Module
Provides text-to-speech functionality for blind users
"""

import pyttsx3
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioOutput:
    """Handles text-to-speech output for product descriptions"""
    
    def __init__(self, rate: int = 150, voice_index: int = 0):
        """
        Initialize audio output
        
        Args:
            rate: Speech rate in words per minute
            voice_index: Index of voice to use
        """
        self.engine = None
        self.rate = rate
        self.voice_index = voice_index
        self.enabled = True
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Set speech rate
            self.engine.setProperty('rate', self.rate)
            
            # Set voice
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > self.voice_index:
                self.engine.setProperty('voice', voices[self.voice_index].id)
            
            logger.info("Audio output initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing audio engine: {str(e)}")
            self.enabled = False
    
    def speak(self, text: str, wait: bool = True):
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            wait: Whether to wait for speech to complete
        """
        if not self.enabled or not self.engine:
            logger.warning("Audio output not available, printing text instead:")
            print(text)
            return
        
        try:
            self.engine.say(text)
            if wait:
                self.engine.runAndWait()
            else:
                self.engine.startLoop(False)
                self.engine.iterate()
                self.engine.endLoop()
                
        except Exception as e:
            logger.error(f"Error speaking text: {str(e)}")
            print(f"[Audio Error] {text}")
    
    def speak_async(self, text: str):
        """Speak text asynchronously (non-blocking)"""
        self.speak(text, wait=False)
    
    def stop(self):
        """Stop current speech"""
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def set_rate(self, rate: int):
        """Set speech rate"""
        self.rate = rate
        if self.engine:
            self.engine.setProperty('rate', rate)
    
    def set_voice(self, voice_index: int):
        """Set voice index"""
        self.voice_index = voice_index
        if self.engine and self.engine.getProperty('voices'):
            voices = self.engine.getProperty('voices')
            if len(voices) > voice_index:
                self.engine.setProperty('voice', voices[voice_index].id)

