# Importing the required libraries
import cv2
import os
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image
import psycopg2
from sklearn.preprocessing import normalize

# Suppressing oneDNN warnings from TensorFlow
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Loading the Haar Cascade algorithm file
alg = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Loading the image path
file_name = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\bigbang.png"
img = cv2.imread(file_name, 0)  # Reading the image
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Creating a black and white version of the image

# Detecting the faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
)

i = 0
# For each detected face, crop the image and save
for x, y, w, h in faces:
    cropped_image = img[y : y + h, x : x + w]  # Cropping the face
    target_file_name = f'stored-faces/{i}.jpg'
    cv2.imwrite(target_file_name, cropped_image)  # Saving the cropped face
    i += 1

# Connecting to the PostgreSQL database
conn = psycopg2.connect("postgres://avnadmin:AVNS_FRtYHG63QjXPdo1jxt2@pg-13195976-faces-nalin.i.aivencloud.com:13387/defaultdb?sslmode=require")

# Initialize imgbeddings model
ibed = imgbeddings()

# Process each cropped face image and insert it into the database
for filename in os.listdir("stored-faces"):
    img = Image.open("stored-faces/" + filename)  # Opening the image
    embedding = ibed.to_embeddings(img)  # Calculating the embeddings
    normalized_embedding = normalize([embedding[0]])[0]  # Normalize the embeddings

    cur = conn.cursor()
    # Check if the record with the same filename already exists
    cur.execute("SELECT * FROM pictures WHERE picture = %s", (filename,))
    result = cur.fetchone()

    if result is None:
        # Insert the new image if it doesn't exist
        cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (filename, normalized_embedding.tolist()))
        print(f"Inserted: {filename}")
    else:
        print(f"Skipped: {filename} (already exists)")

conn.commit()

# Now, let's find and display the most similar face to a new input image
file_name = "C:\\Users\\LENOVO\\OneDrive\\Desktop\\face_rec\\cooper.png"  # Path to the input face image
img = Image.open(file_name)  # Open the image
embedding = ibed.to_embeddings(img)  # Calculate the embeddings for the input image
normalized_embedding = normalize([embedding[0]])[0]  # Normalize the input image embedding

# Query the database for the most similar face
cur = conn.cursor()
string_representation = "[" + ",".join(str(x) for x in normalized_embedding.tolist()) + "]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()

# Function to display the output image
def display_image(image_path):
    img_to_display = cv2.imread(image_path)
    cv2.imshow('Most Similar Face', img_to_display)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

# Show the most similar face
if rows:
    for row in rows:
        image_path = "stored-faces/" + row[0]
        print(f"Displaying: {row[0]}")
        display_image(image_path)

# Closing the cursor and the connection
cur.close()
conn.close()
