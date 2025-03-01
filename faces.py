#!/usr/bin/env python3
"""
Enterprise-Grade Face Recognition System

Features:
- Face detection and embedding generation
- Similarity search and anti-spoofing
- CLI interface for adding, searching, listing, and cleaning up faces
- Data encryption using Fernet
- Optional face clustering and GPU acceleration
"""

import os
import sqlite3
import argparse
import logging
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import face_recognition
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
from mtcnn import MTCNN
from deepface import DeepFace
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from cryptography.fernet import Fernet
from logging.handlers import RotatingFileHandler

# -------------------- Configuration --------------------
class Config:
    """Configuration settings for the face recognition system."""
    DATABASE_PATH = os.getenv("DATABASE_PATH", "face_db.sqlite")
    IMAGE_SIZE = (250, 250)  # Balance between speed and accuracy
    DETECTION_METHOD = os.getenv("DETECTION_METHOD", "mtcnn")  # Options: "haar", "dlib", "mtcnn"
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.6))
    ENCODING_VERSION = os.getenv("ENCODING_VERSION", "Facenet")  # Options: "Facenet", "VGG-Face", "ArcFace"
    ANTI_SPOOFING = os.getenv("ANTI_SPOOFING", "True") == "True"
    LANDMARKS_MODEL = os.getenv("LANDMARKS_MODEL", "large")  # "small" for faster detection
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", 4))  # For parallel processing
    LOG_FILE = os.getenv("LOG_FILE", "face_recognition.log")
    ENABLE_CACHING = os.getenv("ENABLE_CACHING", "True") == "True"  # Cache embeddings for faster searches
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", 30))  # Auto-delete old entries
    ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY", Fernet.generate_key().decode())
    CLOUD_STORAGE_URL = os.getenv("CLOUD_STORAGE_URL")
    ENABLE_CLUSTERING = os.getenv("ENABLE_CLUSTERING", "True") == "True"  # Enable face clustering
    ENABLE_REALTIME = os.getenv("ENABLE_REALTIME", "True") == "True"  # Enable real-time processing
    GPU_ACCELERATION = os.getenv("GPU_ACCELERATION", "True") == "True"  # Enable GPU acceleration

# -------------------- Logging Setup --------------------
def setup_logging() -> logging.Logger:
    """Set up logging with both console and rotating file handlers."""
    logger = logging.getLogger("FaceRecognition")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Rotating file handler (max 5 MB per file, with 3 backups)
    fh = RotatingFileHandler(Config.LOG_FILE, maxBytes=5*1024*1024, backupCount=3)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

logger = setup_logging()

# -------------------- Security --------------------
class SecurityManager:
    """
    Manages encryption and decryption for sensitive data.
    Uses Fernet symmetric encryption.
    """
    def __init__(self):
        self.cipher = Fernet(Config.ENCRYPTION_KEY.encode())

    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypts data using the configured key."""
        try:
            return self.cipher.encrypt(data)
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise

    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypts data using the configured key."""
        try:
            return self.cipher.decrypt(encrypted_data)
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise

# -------------------- Database Management --------------------
class FaceDatabase:
    """
    Handles database operations such as adding faces, searching,
    listing, and cleaning up old entries.
    """
    def __init__(self, db_path: str = Config.DATABASE_PATH):
        self.db_path = db_path
        self.security = SecurityManager()
        self.conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Initializes the faces table and indexes."""
        try:
            with self.conn:
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        embedding BLOB NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        metadata TEXT,
                        UNIQUE(name, embedding)
                    )
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON faces(timestamp)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_name ON faces(name)")
            logger.info("Database initialized successfully.")
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    def add_face(self, name: str, embedding: np.ndarray, metadata: Optional[Dict] = None):
        """Adds a face record to the database."""
        try:
            encrypted_embedding = self.security.encrypt_data(embedding.tobytes())
            with self.conn:
                self.conn.execute(
                    "INSERT INTO faces (name, embedding, metadata) VALUES (?, ?, ?)",
                    (name, encrypted_embedding, json.dumps(metadata) if metadata else None),
                )
            logger.info(f"Added face: {name}")
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate face entry: {name}")
        except Exception as e:
            logger.error(f"Failed to add face: {e}")

    def find_similar(self, embedding: np.ndarray, threshold: float) -> List[Dict]:
        """
        Searches for faces similar to the given embedding.
        Returns records with similarity above the threshold.
        """
        try:
            cursor = self.conn.execute("SELECT id, name, embedding FROM faces")
            target_embedding = normalize([embedding])
            results = []

            for row in cursor:
                try:
                    decrypted_embedding = self.security.decrypt_data(row["embedding"])
                    db_embedding = np.frombuffer(decrypted_embedding, dtype=np.float64)
                    db_embedding = normalize([db_embedding])
                    similarity = float(np.dot(target_embedding, db_embedding.T)[0][0])
                    if similarity >= threshold:
                        results.append({
                            "id": row["id"],
                            "name": row["name"],
                            "similarity": similarity
                        })
                except Exception as de:
                    logger.error(f"Error processing embedding for id {row['id']}: {de}")

            return sorted(results, key=lambda x: x["similarity"], reverse=True)
        except sqlite3.Error as e:
            logger.error(f"Database query error: {e}")
            return []

    def list_faces(self) -> List[Dict]:
        """Returns a list of all faces in the database."""
        try:
            cursor = self.conn.execute("SELECT id, name, timestamp FROM faces")
            return [dict(row) for row in cursor]
        except sqlite3.Error as e:
            logger.error(f"Failed to list faces: {e}")
            return []

    def cleanup_old_entries(self):
        """Deletes entries older than the configured retention period."""
        try:
            with self.conn:
                self.conn.execute("""
                    DELETE FROM faces 
                    WHERE timestamp < datetime('now', ?)
                """, (f"-{Config.DATA_RETENTION_DAYS} days",))
            logger.info("Cleaned up old database entries")
        except sqlite3.Error as e:
            logger.error(f"Cleanup error: {e}")

    def close(self):
        """Closes the database connection."""
        try:
            self.conn.close()
            logger.info("Database connection closed.")
        except sqlite3.Error as e:
            logger.error(f"Error closing database: {e}")

# -------------------- Face Recognition --------------------
class FaceRecognizer:
    """
    Handles face detection, embedding generation, and liveness (anti-spoofing) checks.
    """
    def __init__(self):
        # Choose the face detector based on the configuration
        if Config.DETECTION_METHOD == "mtcnn":
            self.detector = MTCNN()
        else:
            self.detector = None
        self.security = SecurityManager()

    def detect_faces(self, image_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """
        Detects faces in an image and returns their embeddings and bounding boxes.
        Bounding boxes follow the (top, right, bottom, left) convention.
        """
        try:
            image = face_recognition.load_image_file(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if Config.DETECTION_METHOD == "mtcnn":
                locations = self._detect_faces_mtcnn(image)
            else:
                locations = face_recognition.face_locations(image, model=Config.DETECTION_METHOD)

            # Apply anti-spoofing check if enabled
            if Config.ANTI_SPOOFING:
                locations = [loc for loc in locations if self._check_liveness(image, loc)]

            # Generate face embeddings
            face_encodings = face_recognition.face_encodings(
                image, locations, num_jitters=2, model=Config.ENCODING_VERSION
            )
            processed_faces = [normalize([encoding])[0] for encoding in face_encodings]
            return processed_faces, locations
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logger.error(f"Image processing failed: {e}")
            return [], []
        except Exception as e:
            logger.error(f"Unexpected error in detect_faces: {e}")
            return [], []

    def _detect_faces_mtcnn(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detects faces using MTCNN and converts boxes to the expected format."""
        try:
            results = self.detector.detect_faces(image)
            boxes = []
            for result in results:
                x, y, w, h = result.get("box", [0, 0, 0, 0])
                # Convert [x, y, w, h] to (top, right, bottom, left)
                boxes.append((y, x + w, y + h, x))
            return boxes
        except Exception as e:
            logger.error(f"MTCNN detection error: {e}")
            return []

    def _check_liveness(self, image: np.ndarray, location: Tuple[int, int, int, int]) -> bool:
        """
        Checks face liveness using facial landmarks.
        Returns True if the face passes the anti-spoofing check.
        """
        try:
            landmarks_list = face_recognition.face_landmarks(image, [location], model=Config.LANDMARKS_MODEL)
            if not landmarks_list:
                return False
            landmarks = landmarks_list[0]
            left_eye = landmarks.get("left_eye", [])
            right_eye = landmarks.get("right_eye", [])
            mouth = landmarks.get("top_lip", []) + landmarks.get("bottom_lip", [])
            if not left_eye or not right_eye or not mouth:
                return False
            ear = self._eye_aspect_ratio(left_eye + right_eye)
            mar = self._mouth_aspect_ratio(mouth)
            # Tunable thresholds: eyes must be open and mouth mostly closed
            return ear > 0.25 and mar < 0.8
        except Exception as e:
            logger.warning(f"Liveness check failed: {e}")
            return False

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        """Calculates the eye aspect ratio (EAR)."""
        if len(eye_points) < 6:
            return 0.0
        vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (vertical1 + vertical2) / (2.0 * horizontal) if horizontal != 0 else 0.0

    @staticmethod
    def _mouth_aspect_ratio(mouth_points: List[Tuple[int, int]]) -> float:
        """Calculates the mouth aspect ratio (MAR)."""
        if len(mouth_points) < 11:
            return 0.0
        vertical = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[10]))
        horizontal = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
        return vertical / horizontal if horizontal != 0 else 0.0

# -------------------- Advanced Clustering (Optional) --------------------
def perform_face_clustering(embeddings: List[np.ndarray], eps: float = 0.5, min_samples: int = 2):
    """
    Performs clustering on face embeddings using DBSCAN and plots the clusters.
    """
    if not embeddings:
        logger.info("No embeddings provided for clustering.")
        return

    embeddings_matrix = np.vstack(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(embeddings_matrix)
    labels = clustering.labels_

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=embeddings_matrix[:, 0], y=embeddings_matrix[:, 1],
                    hue=labels, palette="viridis", legend="full")
    plt.title("Face Embedding Clusters")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()
    logger.info(f"Clustering completed with {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")

# -------------------- GPU Acceleration Setup --------------------
def setup_gpu():
    """Configures TensorFlow for GPU usage if enabled."""
    if Config.GPU_ACCELERATION:
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                logger.info("GPU acceleration enabled.")
            else:
                logger.info("No GPU devices found, running on CPU.")
        except Exception as e:
            logger.error(f"Error setting up GPU acceleration: {e}")

# -------------------- CLI Interface --------------------
def main():
    """Main CLI handler for various face recognition tasks."""
    setup_gpu()  # Configure GPU if available
    parser = argparse.ArgumentParser(
        description="Advanced Enterprise-Grade Face Recognition System"
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Add face command
    add_parser = subparsers.add_parser("add", help="Add faces to the database")
    add_parser.add_argument("image", help="Path to the image file")
    add_parser.add_argument("--name", required=True, help="Name for the face(s)")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar faces")
    search_parser.add_argument("image", help="Path to the query image")

    # List command
    subparsers.add_parser("list", help="List all faces in the database")

    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up old database entries")

    # Clustering command (if enabled)
    if Config.ENABLE_CLUSTERING:
        cluster_parser = subparsers.add_parser("cluster", help="Perform face clustering")
        cluster_parser.add_argument("image_folder", help="Path to folder containing face images")

    args = parser.parse_args()
    db = FaceDatabase()

    try:
        if args.command == "add":
            recognizer = FaceRecognizer()
            embeddings, locations = recognizer.detect_faces(args.image)
            if not embeddings:
                print("No valid faces found in image")
                return

            for idx, encoding in enumerate(embeddings):
                metadata = {"original_image": args.image, "face_location": locations[idx]}
                db.add_face(args.name, encoding, metadata)
            print(f"Added {len(embeddings)} face(s) to the database.")

        elif args.command == "search":
            recognizer = FaceRecognizer()
            embeddings, _ = recognizer.detect_faces(args.image)
            if not embeddings:
                print("No faces found in query image")
                return

            results = db.find_similar(embeddings[0], Config.SIMILARITY_THRESHOLD)
            if results:
                print("Matching faces:")
                for match in results:
                    print(f"- {match['name']} (similarity: {match['similarity']:.2%})")
            else:
                print("No matches found.")

        elif args.command == "list":
            faces = db.list_faces()
            if faces:
                for face in faces:
                    print(f"ID: {face['id']} | Name: {face['name']} | Added: {face['timestamp']}")
            else:
                print("No faces in the database.")

        elif args.command == "cleanup":
            db.cleanup_old_entries()
            print("Database cleanup completed.")

        elif args.command == "cluster" and Config.ENABLE_CLUSTERING:
            image_folder = args.image_folder
            if not os.path.isdir(image_folder):
                print(f"Invalid folder: {image_folder}")
                return

            embeddings = []
            recognizer = FaceRecognizer()
            images = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                      if os.path.isfile(os.path.join(image_folder, f))]
            with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
                future_to_image = {executor.submit(recognizer.detect_faces, img): img for img in images}
                for future in as_completed(future_to_image):
                    try:
                        emb, _ = future.result()
                        if emb:
                            embeddings.extend(emb)
                    except Exception as e:
                        logger.error(f"Error processing image {future_to_image[future]}: {e}")
            perform_face_clustering(embeddings)
        else:
            parser.print_help()
    finally:
        db.close()

if __name__ == "__main__":
    main()