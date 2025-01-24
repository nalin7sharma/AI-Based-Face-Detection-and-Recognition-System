import os
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
from psycopg2.extras import Json
from sklearn.preprocessing import normalize
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress oneDNN warnings from TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths and constants
HAAR_CASCADE_PATH = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\haarcascade_frontalface_default.xml"
INPUT_IMAGE_PATH = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\bigbang.png"
STORED_FACES_DIR = "stored-faces"
os.makedirs(STORED_FACES_DIR, exist_ok=True)

def load_haar_cascade(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Haar Cascade file not found: {path}")
    return cv2.CascadeClassifier(path)

def detect_and_save_faces(image_path, haar_cascade, output_dir):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

    if len(faces) == 0:
        logging.warning("No faces detected in the image.")
        return 0

    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = img[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (128, 128))  # Resize to a consistent size
        output_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(output_path, cropped_image)
        logging.info(f"Saved face to: {output_path}")

    return len(faces)

def connect_to_database():
    try:
        conn = psycopg2.connect(
            os.environ.get("DB_CONNECTION_STRING"),
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    except Exception as e:
        logging.error("Failed to connect to the database.", exc_info=True)
        raise e

def process_and_store_embeddings(db_conn, imgbeddings_model, directory):
    cursor = db_conn.cursor()
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        img = Image.open(file_path)
        embedding = imgbeddings_model.to_embeddings(img)
        normalized_embedding = normalize([embedding[0]])[0]

        cursor.execute("SELECT * FROM pictures WHERE picture = %s", (filename,))
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO pictures (picture, embedding) VALUES (%s, %s)",
                (filename, Json(normalized_embedding.tolist()))
            )
            logging.info(f"Inserted: {filename}")
        else:
            logging.info(f"Skipped: {filename} (already exists)")
    db_conn.commit()

def find_most_similar_face(db_conn, imgbeddings_model, input_image_path):
    img = Image.open(input_image_path)
    embedding = imgbeddings_model.to_embeddings(img)
    normalized_embedding = normalize([embedding[0]])[0]

    cursor = db_conn.cursor()
    embedding_as_json = Json(normalized_embedding.tolist())
    cursor.execute(
        "SELECT picture, embedding FROM pictures ORDER BY embedding <-> %s LIMIT 1",
        (embedding_as_json,)
    )
    result = cursor.fetchone()
    if result:
        return os.path.join(STORED_FACES_DIR, result["picture"])
    return None

def display_image(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imshow("Most Similar Face", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logging.warning(f"Image not found: {image_path}")

def main():
    try:
        # Load Haar Cascade
        haar_cascade = load_haar_cascade(HAAR_CASCADE_PATH)

        # Detect and save faces
        face_count = detect_and_save_faces(INPUT_IMAGE_PATH, haar_cascade, STORED_FACES_DIR)
        if face_count == 0:
            logging.warning("No faces detected. Exiting.")
            return

        # Connect to the database
        db_conn = connect_to_database()

        # Initialize imgbeddings
        imgbeddings_model = imgbeddings()

        # Process and store embeddings
        process_and_store_embeddings(db_conn, imgbeddings_model, STORED_FACES_DIR)

        # Find the most similar face
        most_similar_image_path = find_most_similar_face(
            db_conn, imgbeddings_model, "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\cooper.png"
        )

        if most_similar_image_path:
            logging.info(f"Displaying most similar face: {most_similar_image_path}")
            display_image(most_similar_image_path)
        else:
            logging.warning("No similar face found in the database.")

    except Exception as e:
        logging.error("An error occurred during execution.", exc_info=True)
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
