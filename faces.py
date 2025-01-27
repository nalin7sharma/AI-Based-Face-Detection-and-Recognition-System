import os
import cv2
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
from psycopg2.extras import Json
from sklearn.preprocessing import normalize
import logging
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress oneDNN warnings from TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths and constants
HAAR_CASCADE_PATH = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\haarcascade_frontalface_default.xml"
STORED_FACES_DIR = "stored-faces"
os.makedirs(STORED_FACES_DIR, exist_ok=True)

# Streamlit setup
st.title("Face Recognition with Similarity Search")
st.write("Upload two images: one to detect faces and another to compare similarity.")

# Load Haar Cascade
def load_haar_cascade(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Haar Cascade file not found: {path}")
    return cv2.CascadeClassifier(path)

# Detect and save faces
def detect_and_save_faces(image, haar_cascade, output_dir):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100))

    if len(faces) == 0:
        st.warning("No faces detected in the image.")
        return 0, None

    face_images = []
    for i, (x, y, w, h) in enumerate(faces):
        cropped_image = image[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (128, 128))  # Resize to a consistent size
        output_path = os.path.join(output_dir, f"{i}.jpg")
        cv2.imwrite(output_path, cropped_image)
        face_images.append(output_path)

    return len(faces), face_images

# Connect to database
def connect_to_database():
    try:
        conn = psycopg2.connect(
            os.environ.get("DB_CONNECTION_STRING"),
            cursor_factory=psycopg2.extras.RealDictCursor
        )
        return conn
    except Exception as e:
        st.error("Failed to connect to the database.")
        raise e

# Process and store embeddings
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

# Find most similar face
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

# Streamlit UI
def main():
    try:
        # Load Haar Cascade
        haar_cascade = load_haar_cascade(HAAR_CASCADE_PATH)

        # Image upload: Image 1 for face detection
        st.header("Step 1: Upload an image for face detection")
        uploaded_image1 = st.file_uploader("Upload Image 1 (for Face Detection)", type=["jpg", "png", "jpeg"])
        
        # Image upload: Image 2 for comparison
        st.header("Step 2: Upload an image for similarity comparison")
        uploaded_image2 = st.file_uploader("Upload Image 2 (for Comparison)", type=["jpg", "png", "jpeg"])

        if uploaded_image1 and uploaded_image2:
            # Read and display Image 1 (Face Detection)
            img1 = np.array(Image.open(uploaded_image1))
            st.image(img1, caption="Uploaded Image 1 (for Face Detection)", use_column_width=True)

            # Detect and save faces from Image 1
            face_count, face_images = detect_and_save_faces(img1, haar_cascade, STORED_FACES_DIR)
            if face_count > 0:
                st.write(f"Detected {face_count} face(s) in Image 1.")

                # Display detected faces
                for face_image in face_images:
                    st.image(face_image, caption=f"Detected Face {face_images.index(face_image)+1}", use_column_width=True)

                # Connect to the database
                db_conn = connect_to_database()

                # Initialize imgbeddings
                imgbeddings_model = imgbeddings()

                # Process and store embeddings of the detected faces
                process_and_store_embeddings(db_conn, imgbeddings_model, STORED_FACES_DIR)

                # Find most similar face based on Image 2 (comparison image)
                img2 = np.array(Image.open(uploaded_image2))
                st.image(img2, caption="Uploaded Image 2 (for Comparison)", use_column_width=True)

                most_similar_image_path = find_most_similar_face(db_conn, imgbeddings_model, uploaded_image2)

                if most_similar_image_path:
                    st.write(f"Most similar face found: {most_similar_image_path}")
                    st.image(most_similar_image_path, caption="Most Similar Face", use_column_width=True)
                else:
                    st.warning("No similar face found in the database.")
            else:
                st.warning("No faces detected in Image 1.")
    
    except Exception as e:
        st.error("An error occurred during execution.")
        logging.error("An error occurred during execution.", exc_info=True)

    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()
            logging.info("Database connection closed.")

if __name__ == "__main__":
    main()
