import os
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image, UnidentifiedImageError
import psycopg2
from psycopg2.extras import DictCursor
from sklearn.preprocessing import normalize
import logging
import streamlit as st
from typing import List, Tuple, Optional
from contextlib import contextmanager
import tempfile
import hashlib
import dlib  # For face alignment
from deepface import DeepFace  # For additional face recognition models
from datetime import datetime

# -------------------- Configuration --------------------
class Config:
    HAAR_CASCADE_PATH = os.getenv("HAAR_CASCADE_PATH", "haarcascade_frontalface_default.xml")
    STORED_FACES_DIR = os.getenv("STORED_FACES_DIR", "stored-faces")
    DB_CONN_STRING = os.getenv("DB_CONN_STRING")
    IMAGE_SIZE = (160, 160)  # Updated for better face recognition
    DETECTION_SCALE_FACTOR = 1.05
    DETECTION_MIN_NEIGHBORS = 5
    DETECTION_MIN_SIZE = (100, 100)
    SIMILARITY_THRESHOLD = 0.8
    FACE_ALIGNMENT = True  # Enable face alignment
    FACE_RECOGNITION_MODEL = "Facenet"  # Options: "Facenet", "VGG-Face", "OpenFace", "DeepFace"

os.makedirs(Config.STORED_FACES_DIR, exist_ok=True)

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Database Handler --------------------
class DatabaseHandler:
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = psycopg2.connect(
                Config.DB_CONN_STRING,
                cursor_factory=DictCursor
            )
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database error: {e}")
            st.error("Database operation failed")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_database(self):
        """Initialize database tables and indexes"""
        with self.get_connection() as conn, conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pictures (
                    id SERIAL PRIMARY KEY,
                    picture TEXT UNIQUE,
                    embedding cube,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS embedding_idx ON pictures USING gist(embedding);
            """)
            conn.commit()

# -------------------- Face Processor --------------------
class FaceProcessor:
    def __init__(self):
        self.haar_cascade = self._load_haar_cascade()
        self.imgbeddings_model = imgbeddings()
        self.face_detector = dlib.get_frontal_face_detector() if Config.FACE_ALIGNMENT else None
        self.face_landmarks = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") if Config.FACE_ALIGNMENT else None

    def _load_haar_cascade(self):
        """Load and verify Haar Cascade classifier"""
        if not os.path.exists(Config.HAAR_CASCADE_PATH):
            raise FileNotFoundError(f"Haar Cascade file not found: {Config.HAAR_CASCADE_PATH}")
        return cv2.CascadeClassifier(Config.HAAR_CASCADE_PATH)

    def _validate_image(self, image: np.ndarray) -> bool:
        """Basic image validation"""
        if image is None or image.size == 0:
            raise ValueError("Invalid image file")
        return True

    def _align_face(self, image: np.ndarray, face_rect) -> np.ndarray:
        """Align face using facial landmarks"""
        landmarks = self.face_landmarks(image, face_rect)
        aligned_face = dlib.get_face_chip(image, landmarks)
        return aligned_face

    def detect_faces(self, image: np.ndarray) -> Tuple[int, List[str]]:
        """Detect faces in an image and save cropped faces"""
        self._validate_image(image)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if Config.FACE_ALIGNMENT:
            faces = self.face_detector(gray_img)
        else:
            faces = self.haar_cascade.detectMultiScale(
                gray_img,
                scaleFactor=Config.DETECTION_SCALE_FACTOR,
                minNeighbors=Config.DETECTION_MIN_NEIGHBORS,
                minSize=Config.DETECTION_MIN_SIZE
            )

        if len(faces) == 0:
            return 0, []

        face_paths = []
        for i, face in enumerate(faces):
            try:
                if Config.FACE_ALIGNMENT:
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    face_img = self._align_face(image, face)
                else:
                    x, y, w, h = face
                    face_img = image[y:y+h, x:x+w]

                face_img = cv2.resize(face_img, Config.IMAGE_SIZE)
                filename = f"face_{hashlib.sha256(face_img.tobytes()).hexdigest()[:16]}.jpg"
                file_path = os.path.join(Config.STORED_FACES_DIR, filename)
                cv2.imwrite(file_path, face_img)
                face_paths.append(file_path)
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")

        return len(faces), face_paths

    def generate_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Generate normalized embedding for an image"""
        try:
            if Config.FACE_RECOGNITION_MODEL == "imgbeddings":
                with Image.open(image_path) as img:
                    embedding = self.imgbeddings_model.to_embeddings(img)
                    return normalize([embedding[0]])[0]
            else:
                embedding = DeepFace.represent(image_path, model_name=Config.FACE_RECOGNITION_MODEL)
                return normalize([embedding[0]["embedding"]])[0]
        except (IOError, UnidentifiedImageError) as e:
            logger.error(f"Invalid image file {image_path}: {e}")
            return None

# -------------------- Streamlit App --------------------
class FaceRecognitionApp:
    def __init__(self):
        self.db_handler = DatabaseHandler()
        self.face_processor = FaceProcessor()
        self.db_handler.initialize_database()

    def _save_embeddings_to_db(self, face_paths: List[str]):
        """Save face embeddings to database"""
        with self.db_handler.get_connection() as conn, conn.cursor() as cursor:
            for path in face_paths:
                filename = os.path.basename(path)
                embedding = self.face_processor.generate_embedding(path)
                
                if embedding is None:
                    continue

                try:
                    cursor.execute(
                        """INSERT INTO pictures (picture, embedding)
                        VALUES (%s, CUBE(%s))
                        ON CONFLICT (picture) DO NOTHING""",
                        (filename, embedding.tolist())
                    )
                    logger.info(f"Processed: {filename}")
                except psycopg2.Error as e:
                    logger.error(f"Database insert error: {e}")

            conn.commit()

    def _find_similar_face(self, query_image_path: str) -> Optional[str]:
        """Find most similar face in database"""
        query_embedding = self.face_processor.generate_embedding(query_image_path)
        if query_embedding is None:
            return None

        with self.db_handler.get_connection() as conn, conn.cursor() as cursor:
            try:
                cursor.execute(
                    """SELECT picture, 
                       1 - (embedding <-> CUBE(%s)) as similarity 
                       FROM pictures 
                       WHERE 1 - (embedding <-> CUBE(%s)) > %s
                       ORDER BY similarity DESC 
                       LIMIT 1""",
                    (query_embedding.tolist(), 
                     query_embedding.tolist(), 
                     Config.SIMILARITY_THRESHOLD)
                )
                result = cursor.fetchone()
                return os.path.join(Config.STORED_FACES_DIR, result["picture"]) if result else None
            except psycopg2.Error as e:
                logger.error(f"Database query error: {e}")
                return None

    def run(self):
        """Main application flow"""
        st.title("Face Recognition with Similarity Search")
        st.markdown("""
            **Instructions:**
            1. Upload a reference image for face detection
            2. Upload a query image for similarity search
            3. View detected faces and similarity results
        """)

        with st.expander("Advanced Settings"):
            st.write(f"Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
            st.write(f"Face Recognition Model: {Config.FACE_RECOGNITION_MODEL}")
            st.write(f"Face Alignment: {'Enabled' if Config.FACE_ALIGNMENT else 'Disabled'}")

        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Reference Image")
            ref_file = st.file_uploader("Upload reference image", type=["jpg", "png", "jpeg"])

        with col2:
            st.header("Query Image")
            query_file = st.file_uploader("Upload query image", type=["jpg", "png", "jpeg"])

        if ref_file and query_file:
            try:
                # Process reference image
                with st.spinner("Processing reference image..."):
                    ref_image = np.array(Image.open(ref_file))
                    st.image(ref_image, caption="Reference Image", use_column_width=True)
                    
                    face_count, face_paths = self.face_processor.detect_faces(ref_image)
                    if face_count == 0:
                        st.warning("No faces detected in reference image")
                        return
                    
                    st.success(f"Detected {face_count} face(s)")
                    self._save_embeddings_to_db(face_paths)

                    # Display detected faces
                    st.subheader("Detected Faces")
                    cols = st.columns(min(3, face_count))
                    for idx, (col, path) in enumerate(zip(cols, face_paths)):
                        with col:
                            st.image(path, caption=f"Face {idx+1}", use_column_width=True)

                # Process query image
                with st.spinner("Searching for similar faces..."):
                    query_image = np.array(Image.open(query_file))
                    st.image(query_image, caption="Query Image", use_column_width=True)
                    
                    # Save query image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                        Image.fromarray(query_image).save(tmp_file.name)
                        similar_path = self._find_similar_face(tmp_file.name)
                    
                    if similar_path:
                        st.success("Similar face found!")
                        st.image(similar_path, caption="Most Similar Face", use_column_width=True)
                    else:
                        st.warning("No similar faces found above threshold")

            except Exception as e:
                logger.error(f"Application error: {e}", exc_info=True)
                st.error("An error occurred during processing")

# -------------------- Main Execution --------------------
if __name__ == "__main__":
    # Suppress TensorFlow logging
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    
    # Initialize and run application
    try:
        app = FaceRecognitionApp()
        app.run()
    except Exception as e:
        st.error("Critical application error. Please check logs.")
        logger.critical(f"Application failed: {e}", exc_info=True)
