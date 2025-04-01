#!/usr/bin/env python3
"""
Enterprise-Grade Face Recognition System 2.0

Key Upgrades:
- Microservices architecture with REST API
- PostgreSQL + pgvector for scalable similarity search
- Advanced anti-spoofing with deep learning
- JWT authentication
- Async processing
- Enhanced monitoring
- Model versioning
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

import cv2
import numpy as np
import asyncpg
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, BaseSettings, validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from sklearn.preprocessing import normalize
from deepface import DeepFace
from deepface.modules import verification
from mtcnn import MTCNN
from prometheus_fastapi_instrumentator import Instrumentator
from loguru import logger

# -------------------- Configuration --------------------
class Settings(BaseSettings):
    POSTGRES_URL: str = "postgresql://user:pass@localhost:5432/facedb"
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    MODEL_VERSION: str = "Facenet512"
    DETECTOR_BACKEND: str = "mtcnn"
    SIMILARITY_THRESHOLD: float = 0.67
    ANTI_SPOOFING: bool = True
    ENABLE_METRICS: bool = True
    
    class Config:
        env_file = ".env"

settings = Settings()

# -------------------- Security --------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# -------------------- Database Models --------------------
class FaceBase(BaseModel):
    user_id: uuid.UUID
    embedding: List[float]
    model_version: str
    metadata: Dict[str, Any] = {}

class FaceCreate(FaceBase):
    password: str  # For demo purposes only

class Face(FaceBase):
    id: uuid.UUID
    created_at: datetime

    class Config:
        orm_mode = True

# -------------------- Services --------------------
class DatabaseService:
    def __init__(self):
        self.pool = None

    async def connect(self):
        self.pool = await asyncpg.create_pool(
            dsn=settings.POSTGRES_URL,
            min_size=5,
            max_size=20
        )
        await self._init_db()

    async def _init_db(self):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS faces (
                    id UUID PRIMARY KEY,
                    user_id UUID NOT NULL,
                    embedding vector(512) NOT NULL,
                    model_version TEXT NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                );
                CREATE INDEX ON faces USING ivfflat (embedding vector_l2_ops);
            """)

    @contextmanager
    def metrics_connection(self):
        conn = self.pool.acquire()
        try:
            yield conn
        finally:
            self.pool.release(conn)

class FaceService:
    def __init__(self, db: DatabaseService):
        self.db = db
        self.detector = MTCNN()
        self.anti_spoof_model = DeepFace.build_model("DeepFake")

    async def detect_faces(self, image_path: str):
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Face detection
            faces = self.detector.detect_faces(img_rgb)
            if not faces:
                return []

            # Anti-spoofing
            valid_faces = []
            for face in faces:
                if settings.ANTI_SPOOFING:
                    if not await self.check_liveness(img_rgb, face):
                        continue
                valid_faces.append(face)

            # Embedding generation
            embeddings = []
            for face in valid_faces:
                x, y, w, h = face['box']
                face_img = img_rgb[y:y+h, x:x+w]
                embedding = DeepFace.represent(
                    face_img,
                    model_name=settings.MODEL_VERSION,
                    enforce_detection=False
                )
                embeddings.append(normalize([embedding]).tolist()[0])

            return embeddings
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            raise

    async def check_liveness(self, image: np.ndarray, face: dict) -> bool:
        try:
            x, y, w, h = face['box']
            face_roi = image[y:y+h, x:x+w]
            prediction = DeepFace.analyze(
                face_roi,
                actions=["deepfake"],
                detector_backend="skip",
                models={"deepfake": self.anti_spoof_model},
                enforce_detection=False
            )
            return prediction["deepfake"]["real"] > 0.8
        except Exception as e:
            logger.warning(f"Liveness check failed: {e}")
            return False

class AuthService:
    def __init__(self, db: DatabaseService):
        self.db = db

    async def authenticate_user(self, username: str, password: str):
        async with self.db.pool.acquire() as conn:
            user = await conn.fetchrow(
                "SELECT * FROM users WHERE username = $1", username
            )
            if not user:
                return False
            if not pwd_context.verify(password, user["password"]):
                return False
            return user

    def create_access_token(self, data: dict):
        expires = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
        data.update({"exp": expires})
        return jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

# -------------------- API Endpoints --------------------
app = FastAPI(title="Face Recognition API")

@app.on_event("startup")
async def startup():
    db = DatabaseService()
    await db.connect()
    app.state.db = db
    app.state.face_service = FaceService(db)
    app.state.auth_service = AuthService(db)
    
    if settings.ENABLE_METRICS:
        Instrumentator().instrument(app).expose(app)

# Dependency
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    async with app.state.db.pool.acquire() as conn:
        user = await conn.fetchrow(
            "SELECT * FROM users WHERE username = $1", username
        )
        if user is None:
            raise credentials_exception
        return user

@app.post("/faces/", response_model=Face)
async def create_face(face: FaceCreate, current_user: dict = Depends(get_current_user)):
    face_service = app.state.face_service
    embeddings = await face_service.detect_faces(face.image_path)
    
    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid faces detected")
    
    async with app.state.db.pool.acquire() as conn:
        face_id = uuid.uuid4()
        await conn.execute("""
            INSERT INTO faces (id, user_id, embedding, model_version, metadata)
            VALUES ($1, $2, $3, $4, $5)
        """, face_id, face.user_id, embeddings[0], settings.MODEL_VERSION, face.metadata)
    
    return {**face.dict(), "id": face_id, "created_at": datetime.now()}

@app.get("/faces/search")
async def search_faces(image_path: str, threshold: float = None):
    face_service = app.state.face_service
    threshold = threshold or settings.SIMILARITY_THRESHOLD
    
    embeddings = await face_service.detect_faces(image_path)
    if not embeddings:
        return {"matches": []}
    
    async with app.state.db.pool.acquire() as conn:
        results = await conn.fetch("""
            SELECT id, user_id, metadata, embedding <-> $1 as distance
            FROM faces
            WHERE embedding <-> $1 < $2
            ORDER BY distance
            LIMIT 10
        """, embeddings[0], threshold)
    
    return {
        "matches": [
            {
                "id": str(record["id"]),
                "user_id": str(record["user_id"]),
                "distance": float(record["distance"]),
                "metadata": record["metadata"]
            }
            for record in results
        ]
    }

@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends())):
    user = await app.state.auth_service.authenticate_user(
        form_data.username, form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = app.state.auth_service.create_access_token(
        data={"sub": user["username"]}
    )
    return {"access_token": access_token, "token_type": "bearer"}

# -------------------- Monitoring --------------------
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# -------------------- Main --------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
