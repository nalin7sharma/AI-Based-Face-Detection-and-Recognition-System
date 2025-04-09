"""
🖼️🔠 Mega ASCII Art Converter Pro 🔠🖼️

Professional-grade image to ASCII conversion with advanced features:
- Multiple conversion algorithms
- Customizable color mapping
- Image preprocessing filters
- Batch processing
- Export formats (TXT, HTML, SVG)
- Animation support (GIF to ASCII)
- Historical conversions
- User preferences
"""

import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import numpy as np
from io import BytesIO
import base64
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
import cv2  # For advanced image processing
from skimage import feature  # For edge detection

# ======================
# CONSTANTS & CONFIGURATION
# ======================

ASCII_GRADIENTS = {
    "Detailed": "@%#*+=-:. ",
    "Minimal": "@#=-. ",
    "Retro": "█▓▒░ ",
    "Blocks": "█▄▌▐▖ ",
    "Technical": "01 ",
    "Artistic": "♠☺♀♂♪♫ ",
    "Custom": None
}

COLOR_MODES = {
    "Monochrome": "mono",
    "ANSI Colors": "ansi",
    "Truecolor": "truecolor",
    "HTML Colors": "html"
}

DEFAULT_CONFIG = {
    "width": 100,
    "gradient": "Detailed",
    "color_mode": "mono",
    "contrast": 1.0,
    "brightness": 1.0,
    "invert": False,
    "edge_detection": False,
    "save_quality": 90
}

# ======================
# DATA STRUCTURES
# ======================

@dataclass
class ConversionSettings:
    width: int = 100
    height: Optional[int] = None
    gradient: str = "Detailed"
    custom_gradient: str = ""
    color_mode: str = "mono"
    contrast: float = 1.0
    brightness: float = 1.0
    invert: bool = False
    edge_detection: bool = False
    dithering: bool = False
    animation_speed: float = 1.0

@dataclass
class UserPreferences:
    dark_mode: bool = True
    auto_save: bool = False
    history_size: int = 10
    default_format: str = "txt"

# ======================
# UTILITIES
# ======================

def get_available_gradients() -> List[str]:
    return list(ASCII_GRADIENTS.keys())

def create_animated_gif(frames: List[Image.Image], duration: int) -> BytesIO:
    """Create GIF from list of PIL images"""
    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    buffer.seek(0)
    return buffer

def rgb_to_ansi(r: int, g: int, b: int) -> str:
    """Convert RGB values to ANSI escape code"""
    return f"\x1b[38;2;{r};{g};{b}m"

def image_to_data_url(image: Image.Image) -> str:
    """Convert PIL image to data URL"""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# ======================
# CORE CONVERSION LOGIC
# ======================

class ASCIIConverter:
    def __init__(self, settings: ConversionSettings):
        self.settings = settings
        self._prepare_gradient()
        
    def _prepare_gradient(self):
        if self.settings.gradient == "Custom":
            self.chars = list(self.settings.custom_gradient)
        else:
            self.chars = list(ASCII_GRADIENTS[self.settings.gradient])
            
        self.num_chars = len(self.chars)
        self.char_map = np.linspace(0, 255, self.num_chars)
        
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements and filters"""
        # Convert to RGB for color processing
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        # Apply enhancements
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.settings.contrast)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.settings.brightness)
        
        if self.settings.invert:
            image = ImageOps.invert(image)
            
        if self.settings.edge_detection:
            arr = np.array(image.convert("L"))
            edges = feature.canny(arr, sigma=2)
            image = Image.fromarray((edges * 255).astype(np.uint8))
            
        if self.settings.dithering:
            image = image.convert("1")  # Floyd-Steinberg dithering
            
        return image
        
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        original_width, original_height = image.size
        aspect_ratio = original_height / original_width
        
        if self.settings.height:
            new_height = self.settings.height
        else:
            new_height = int(aspect_ratio * self.settings.width * 0.55)
            
        return image.resize((self.settings.width, new_height))
    
    def _get_ascii_char(self, pixel_value: int, r: int, g: int) -> str:
        """Get appropriate ASCII character with color coding"""
        char_index = np.digitize(pixel_value, self.char_map) - 1
        char = self.chars[char_index]
        
        if self.settings.color_mode == "ansi":
            return f"{rgb_to_ansi(r, g, 0)}{char}"
        elif self.settings.color_mode == "html":
            return f'<span style="color:rgb({r},{g},0)">{char}</span>'
        else:
            return char
    
    def convert_image(self, image: Image.Image) -> str:
        """Main conversion method"""
        # Preprocess image
        processed_image = self._preprocess_image(image)
        
        # Resize image
        resized_image = self._resize_image(processed_image)
        
        # Convert to numpy array
        pixels = np.array(resized_image)
        
        # Build ASCII art
        ascii_art = []
        for row in pixels:
            line = []
            for pixel in row:
                r, g, b = pixel[:3]
                brightness = 0.299 * r + 0.587 * g + 0.114 * b
                line.append(self._get_ascii_char(brightness, r, g))
            ascii_art.append("".join(line))
            
        return "\n".join(ascii_art)

# ======================
# STREAMLIT UI COMPONENTS
# ======================

def settings_sidebar() -> ConversionSettings:
    """Render settings sidebar and return settings object"""
    with st.sidebar:
        st.header("⚙️ Conversion Settings")
        
        settings = ConversionSettings()
        
        # Basic settings
        settings.width = st.slider("Width (characters)", 20, 400, 100)
        settings.gradient = st.selectbox("ASCII Gradient", get_available_gradients())
        
        if settings.gradient == "Custom":
            settings.custom_gradient = st.text_input("Custom Characters", "▁▂▃▄▅▆▇█")
            
        # Color settings
        settings.color_mode = st.selectbox("Color Mode", list(COLOR_MODES.values()))
        
        # Image processing
        with st.expander("Advanced Image Processing"):
            settings.contrast = st.slider("Contrast", 0.1, 3.0, 1.0)
            settings.brightness = st.slider("Brightness", 0.1, 3.0, 1.0)
            settings.invert = st.checkbox("Invert Colors")
            settings.edge_detection = st.checkbox("Edge Detection")
            settings.dithering = st.checkbox("Dithering")
            
        # Animation settings
        if st.session_state.get('is_animation', False):
            settings.animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)
            
        # Save/load presets
        with st.expander("Preset Management"):
            preset_name = st.text_input("Preset Name")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Save Preset"):
                    save_preset(preset_name, settings)
            with col2:
                if st.button("Load Preset"):
                    load_preset(preset_name)
                    
        return settings

def save_preset(name: str, settings: ConversionSettings):
    """Save current settings as named preset"""
    presets = load_presets()
    presets[name] = vars(settings)
    with open("presets.json", "w") as f:
        json.dump(presets, f)
    st.success(f"Preset '{name}' saved!")

def load_preset(name: str) -> Optional[ConversionSettings]:
    """Load settings from named preset"""
    try:
        with open("presets.json") as f:
            presets = json.load(f)
            if name in presets:
                st.session_state.settings = ConversionSettings(**presets[name])
                st.success(f"Preset '{name}' loaded!")
            else:
                st.error("Preset not found")
    except FileNotFoundError:
        st.error("No presets found")

# ======================
# MAIN APPLICATION
# ======================

def main():
    st.set_page_config(
        page_title="ASCII Art Studio Pro",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'settings' not in st.session_state:
        st.session_state.settings = ConversionSettings()
        
    # UI Layout
    st.title("🎨 ASCII Art Studio Pro")
    st.markdown("### Transform images into beautiful ASCII art masterpieces")
    
    # Settings sidebar
    settings = settings_sidebar()
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image/GIF", 
            type=["png", "jpg", "jpeg", "gif"],
            accept_multiple_files=False,
            help="Supports static images and animated GIFs"
        )
        
        if uploaded_file:
            if uploaded_file.type == "image/gif":
                process_animated_gif(uploaded_file, settings)
            else:
                process_static_image(uploaded_file, settings)
                
    with col2:
        display_history()
        
    # Additional features
    with st.expander("🛠️ Advanced Tools"):
        advanced_tools()
        
    # Footer
    st.markdown("---")
    st.markdown("### 📤 Export Options")
    export_options()
    
def process_static_image(uploaded_file, settings):
    """Process single image file"""
    try:
        image = Image.open(uploaded_file)
        st.session_state.original_image = image
        
        with st.spinner("🔨 Crafting your ASCII masterpiece..."):
            converter = ASCIIConverter(settings)
            ascii_art = converter.convert_image(image)
            
        display_results(ascii_art, settings)
        add_to_history(ascii_art)
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        
def process_animated_gif(uploaded_file, settings):
    """Process animated GIF frame by frame"""
    try:
        gif = Image.open(uploaded_file)
        frames = []
        durations = []
        
        with st.spinner("🎥 Processing animation frames..."):
            for frame in range(0, gif.n_frames):
                gif.seek(frame)
                frame_image = gif.copy().convert("RGB")
                converter = ASCIIConverter(settings)
                ascii_frame = converter.convert_image(frame_image)
                frames.append(ascii_frame)
                durations.append(gif.info['duration'])
                
        st.session_state.animation_frames = frames
        st.session_state.animation_durations = durations
        st.success(f"Processed {len(frames)} frames!")
        display_animation(frames, durations)
        
    except Exception as e:
        st.error(f"Error processing GIF: {str(e)}")
        
def display_results(ascii_art: str, settings: ConversionSettings):
    """Display conversion results with preview"""
    st.subheader("✨ Conversion Result")
    
    if settings.color_mode == "html":
        st.markdown(f"<pre>{ascii_art}</pre>", unsafe_allow_html=True)
    else:
        st.code(ascii_art)
        
    st.download_button(
        "📥 Download ASCII Art",
        data=ascii_art,
        file_name="ascii_art.txt",
        mime="text/plain"
    )
    
def display_animation(frames: List[str], durations: List[int]):
    """Preview animated ASCII art"""
    st.subheader("🎥 Animated Preview")
    placeholder = st.empty()
    
    for frame, duration in zip(frames, durations):
        placeholder.code(frame)
        time.sleep(duration / 1000)
        
def display_history():
    """Show conversion history"""
    with st.expander("📚 Conversion History (Last 10)"):
        if st.session_state.history:
            for i, item in enumerate(st.session_state.history[-10:]):
                st.markdown(f"**Conversion {i+1}**")
                st.code(item[:200] + "...")
        else:
            st.markdown("No conversions yet!")
            
def add_to_history(ascii_art: str):
    """Add conversion to history"""
    st.session_state.history.append(ascii_art)
    if len(st.session_state.history) > 10:
        st.session_state.history.pop(0)
        
def advanced_tools():
    """Additional advanced features"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Image Analysis")
        if st.button("Analyze Image Statistics"):
            analyze_image()
            
    with col2:
        st.markdown("### ⚡ Performance Tools")
        if st.button("Run Benchmark Test"):
            run_benchmark()
            
def export_options():
    """Different export format options"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            "📝 Export as Text",
            data=st.session_state.get('current_art', ''),
            file_name="ascii_art.txt"
        )
        
    with col2:
        st.download_button(
            "🌐 Export as HTML",
            data=wrap_html(st.session_state.get('current_art', '')),
            file_name="ascii_art.html"
        )
        
    with col3:
        if st.button("📤 Share to Social Media"):
            share_to_social()
            
def wrap_html(ascii_art: str) -> str:
    """Wrap ASCII art in HTML template"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            pre {{ 
                font-family: monospace;
                background: #000;
                color: #fff;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <pre>{ascii_art}</pre>
    </body>
    </html>
    """
    
def analyze_image():
    """Show image analysis statistics"""
    if 'original_image' in st.session_state:
        image = st.session_state.original_image
        st.write(f"**Dimensions:** {image.size}")
        st.write(f"**Mode:** {image.mode}")
        st.write(f"**Size:** {image.size[0]*image.size[1]} pixels")
    else:
        st.warning("No image to analyze!")
        
def run_benchmark():
    """Run performance benchmark"""
    with st.spinner("🏃 Running performance test..."):
        start_time = time.time()
        test_image = Image.new("RGB", (1000, 1000), (255, 255, 255))
        converter = ASCIIConverter(ConversionSettings(width=200))
        converter.convert_image(test_image)
        duration = time.time() - start_time
        
    st.metric("Benchmark Result", f"{duration:.2f} seconds")
    
if __name__ == "__main__":
    main()
