
# Face Detection and Recognition System

This project demonstrates a pipeline for detecting faces in an image, storing their embeddings in a PostgreSQL database, and finding the most similar face to a query image using the **imgbeddings** library. The system combines computer vision techniques and database queries to achieve efficient face recognition.

---

## Features

- **Face Detection**: Uses Haar Cascade to detect faces in a given image.
- **Face Cropping and Saving**: Crops detected faces and saves them locally.
- **Embeddings Generation**: Generates normalized embeddings for each cropped face using the `imgbeddings` library.
- **Database Integration**: Stores embeddings in a PostgreSQL database and retrieves similar faces using efficient similarity search.
- **Visualization**: Displays the most similar face to a query image.

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or later
- OpenCV
- Pillow
- numpy
- scikit-learn
- imgbeddings library
- psycopg2 (PostgreSQL adapter for Python)

---

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nalin7sharma/face-recognition.git
   cd face_recognition_project
   ```

2. **Install Dependencies**
   Install the required Python packages:
   ```bash
   pip install opencv-python-headless pillow psycopg2-binary scikit-learn imgbeddings
   ```

3. **Prepare Haar Cascade**
   Download the `haarcascade_frontalface_default.xml` file from the OpenCV repository and update the file path in the code.

4. **Setup Database**
   - Create a PostgreSQL database.
   - Update the `conn` connection string with your database credentials.

5. **Organize Directories**
   Create the following directory for storing cropped face images:
   ```bash
   mkdir stored-faces
   ```

---

## How to Run

### Detect Faces and Store Embeddings
1. Update the `file_name` variable in the code to the path of the input image containing faces.
2. Run the script to detect, crop, and save faces. The embeddings will be stored in the database.

### Find the Most Similar Face
1. Update the `file_name` variable to the path of the query image.
2. Run the script to display the most similar face found in the database.

---

## Code Overview

### 1. **Face Detection**
Detects faces using Haar Cascade and crops them.
```python
haar_cascade = cv2.CascadeClassifier(alg)
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=2, minSize=(100, 100)
)
```

### 2. **Embeddings Generation**
Generates embeddings for each cropped face using the `imgbeddings` library.
```python
embedding = ibed.to_embeddings(img)
normalized_embedding = normalize([embedding[0]])[0]
```

### 3. **Database Integration**
Stores embeddings and retrieves the most similar face using PostgreSQL.
```python
cur.execute("INSERT INTO pictures (picture, embedding) VALUES (%s, %s)", (filename, normalized_embedding.tolist()))
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
```

---

## Acknowledgements

- [OpenCV Haar Cascade](https://github.com/opencv/opencv)
- [imgbeddings Library](https://github.com/ranahanocka/imgbeddings)

---

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.
