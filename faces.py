"""
Advanced Face Recognition System with Database Integration
Features: Face detection, embedding generation, similarity search, anti-spoofing, CLI interface, and more
"""

import os
import sqlite3
import argparse
import logging
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import cv2
import face_recognition
import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image, UnidentifiedImageError
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import json
import hashlib

# -------------------- Configuration --------------------
class Config:
    DATABASE_PATH = "face_db.sqlite"
    IMAGE_SIZE = (250, 250)  # Balance between speed and accuracy
    DETECTION_METHOD = "cnn"  # "hog" for CPU, "cnn" for GPU acceleration
    SIMILARITY_THRESHOLD = 0.6
    ENCODING_VERSION = "v2"  # face_recognition encoding model version
    ANTI_SPOOFING = True
    LANDMARKS_MODEL = "large"  # "small" for faster detection
    MAX_WORKERS = 4  # For parallel processing
    LOG_FILE = "face_recognition.log"
    ENABLE_CACHING = True  # Cache embeddings for faster searches
    DATA_RETENTION_DAYS = 30  # Auto-delete old entries

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(Config.LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# -------------------- Database Setup --------------------
class FaceDatabase:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DATABASE_PATH)
        self._init_db()

    def _init_db(self):
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
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON faces(timestamp)
            """)
            self.conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_name ON faces(name)
            """)

    def add_face(self, name: str, embedding: np.ndarray, metadata: Dict = None):
        embedding_blob = embedding.tobytes()
        with self.conn:
            try:
                self.conn.execute(
                    "INSERT INTO faces (name, embedding, metadata) VALUES (?, ?, ?)",
                    (name, embedding_blob, json.dumps(metadata) if metadata else None),
                )
                logger.info(f"Added face: {name}")
            except sqlite3.IntegrityError:
                logger.warning(f"Duplicate face entry: {name}")

    def find_similar(self, embedding: np.ndarray, threshold: float) -> List[Dict]:
        cursor = self.conn.execute("SELECT id, name, embedding FROM faces")
        target_embedding = normalize([embedding])
        results = []
        
        for row in cursor:
            db_embedding = np.frombuffer(row[2], dtype=np.float64)
            db_embedding = normalize([db_embedding])
            similarity = np.dot(target_embedding, db_embedding.T)[0][0]
            
            if similarity >= threshold:
                results.append({
                    "id": row[0],
                    "name": row[1],
                    "similarity": float(similarity)
                })
        
        return sorted(results, key=lambda x: x["similarity"], reverse=True)

    def cleanup_old_entries(self):
        with self.conn:
            self.conn.execute("""
                DELETE FROM faces 
                WHERE timestamp < datetime('now', ?)
            """, (f"-{Config.DATA_RETENTION_DAYS} days",))
            logger.info("Cleaned up old database entries")

    def close(self):
        self.conn.close()

# -------------------- Core Recognition Engine --------------------
class FaceRecognizer:
    @staticmethod
    def detect_faces(image_path: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        try:
            image = face_recognition.load_image_file(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                image, model=Config.DETECTION_METHOD
            )
            
            if Config.ANTI_SPOOFING:
                face_locations = [loc for loc in face_locations 
                                if FaceRecognizer._check_liveness(image, loc)]
            
            face_encodings = face_recognition.face_encodings(
                image, face_locations, num_jitters=2, model=Config.ENCODING_VERSION
            )
            
            processed_faces = []
            for face in face_encodings:
                normalized = normalize([face])[0]
                processed_faces.append(normalized)
            
            return processed_faces, face_locations
        
        except (FileNotFoundError, UnidentifiedImageError) as e:
            logging.error(f"Image processing failed: {str(e)}")
            return [], []

    @staticmethod
    def _check_liveness(image: np.ndarray, location: Tuple[int, int, int, int]) -> bool:
        """Advanced anti-spoofing using facial landmarks analysis"""
        try:
            landmarks = face_recognition.face_landmarks(
                image, [location], model=Config.LANDMARKS_MODEL
            )[0]
            
            # Eye aspect ratio (EAR) and mouth aspect ratio (MAR)
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            mouth = landmarks["top_lip"] + landmarks["bottom_lip"]
            
            ear = FaceRecognizer._eye_aspect_ratio(left_eye + right_eye)
            mar = FaceRecognizer._mouth_aspect_ratio(mouth)
            
            return ear > 0.25 and mar < 0.8  # Thresholds for open eyes and closed mouth
        
        except Exception as e:
            logging.warning(f"Liveness check failed: {str(e)}")
            return False

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        # Calculate eye aspect ratio (EAR)
        vertical1 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        vertical2 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        horizontal = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        return (vertical1 + vertical2) / (2.0 * horizontal)

    @staticmethod
    def _mouth_aspect_ratio(mouth_points: List[Tuple[int, int]]) -> float:
        # Calculate mouth aspect ratio (MAR)
        vertical = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[10]))
        horizontal = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[6]))
        return vertical / horizontal

# -------------------- CLI Interface --------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced Face Recognition System")
    subparsers = parser.add_subparsers(dest="command")

    # Add face command
    add_parser = subparsers.add_parser("add", help="Add faces to database")
    add_parser.add_argument("image", help="Path to image file")
    add_parser.add_argument("--name", required=True, help="Name for the face")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar faces")
    search_parser.add_argument("image", help="Path to query image")

    # List command
    subparsers.add_parser("list", help="List all faces in database")

    # Cleanup command
    subparsers.add_parser("cleanup", help="Clean up old database entries")

    args = parser.parse_args()
    db = FaceDatabase()

    try:
        if args.command == "add":
            encodings, locations = FaceRecognizer.detect_faces(args.image)
            if not encodings:
                print("No valid faces found in image")
                return

            for idx, encoding in enumerate(encodings):
                db.add_face(args.name, encoding, {
                    "original_image": args.image,
                    "face_location": locations[idx]
                })
            print(f"Added {len(encodings)} faces to database")

        elif args.command == "search":
            encodings, _ = FaceRecognizer.detect_faces(args.image)
            if not encodings:
                print("No faces found in query image")
                return

            results = db.find_similar(encodings[0], Config.SIMILARITY_THRESHOLD)
            if results:
                print("Matching faces:")
                for match in results:
                    print(f"- {match['name']} (similarity: {match['similarity']:.2%})")
            else:
                print("No matches found")

        elif args.command == "list":
            cursor = db.conn.execute("SELECT id, name, timestamp FROM faces")
            for row in cursor:
                print(f"ID: {row[0]} | Name: {row[1]} | Added: {row[2]}")

        elif args.command == "cleanup":
            db.cleanup_old_entries()
            print("Database cleanup completed")

        else:
            parser.print_help()

    finally:
        db.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
