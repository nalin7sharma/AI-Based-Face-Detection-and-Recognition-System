
# Face Recognition with Similarity Search

This project is a face recognition system that allows the user to upload two images: one for face detection and another for comparing similarity with previously stored faces. It leverages Haar Cascade for face detection, `imgeddings` for face embeddings, and a PostgreSQL database to store and search face embeddings.

## Requirements

To run the project, ensure the following dependencies are installed:

1. `streamlit` - For building the web app interface.
2. `opencv-python` - For Haar Cascade face detection.
3. `imgeddings` - For generating face embeddings.
4. `psycopg2` - For PostgreSQL database interaction.
5. `numpy` - For numerical operations.
6. `Pillow` - For image manipulation.
7. `scikit-learn` - For normalizing embeddings.

You can install these dependencies using pip:

```bash
pip install streamlit opencv-python imgbeddings psycopg2 numpy Pillow scikit-learn
```

## Setup

1. Download the Haar Cascade file `haarcascade_frontalface_default.xml` from [here](https://github.com/opencv/opencv/tree/master/data/haarcascades).
2. Set the `HAAR_CASCADE_PATH` variable in the code to the location where you saved the Haar Cascade file.

Make sure you have access to a PostgreSQL database with the necessary schema:

```sql
CREATE TABLE pictures (
    picture TEXT,
    embedding JSONB
);
```

## Running the Application

1. Place the `app.py` file (containing the code provided) in your desired directory.
2. Open a terminal and navigate to that directory.
3. Run the Streamlit app with the following command:

```bash
streamlit run app.py
```

4. Open the provided URL in your browser.
5. Upload two images:
    - **Image 1**: For face detection.
    - **Image 2**: For similarity comparison with the detected faces from Image 1.

## How It Works

- **Step 1**: The user uploads **Image 1**, which is processed for face detection using the Haar Cascade.
- **Step 2**: The faces detected in Image 1 are cropped and stored.
- **Step 3**: The embeddings of the cropped faces are computed and stored in the PostgreSQL database.
- **Step 4**: The user uploads **Image 2**, and the system compares it with the stored faces to find the most similar one.
- **Step 5**: The most similar face is displayed, if found.

## Notes

- Ensure that you have the PostgreSQL database running and the connection string set up in the `DB_CONNECTION_STRING` environment variable.
- If you don't have a database set up, make sure to modify the code to either use an in-memory database or set up a local PostgreSQL instance.

## Usage Example
```bash
# Add faces to database
python main.py add --name "John Doe" john.jpg

# Search for similar faces
python main.py search query.jpg

# List all entries
python main.py list
