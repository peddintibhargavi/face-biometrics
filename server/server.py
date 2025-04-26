from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import os
import sys
import uuid
import pickle
from typing import Dict, List, Optional
import uvicorn
from pydantic import BaseModel
import glob
from pymongo import MongoClient
import base64
from io import BytesIO

# Add the directory containing the modules to the Python path
MODULE_DIR = os.path.dirname(__file__)
sys.path.insert(0, MODULE_DIR)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../FaceUnimodel'))
from feature_conv import float_to_q1_8, q1_8_to_float, secure_enrollment, secure_decrypt, generate_octets
from dot_comapre import distribute_template_shares, secure_and_masked_dot_product, compute_hamming_distance_bits, apply_correction
from detect_noise import resnet_50, Tr
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../common'))
from one_parameter_defense import one_parameter_defense
app = FastAPI(title="Face Biometric Authentication API")
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Serve static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
DATABASE_NAME = "face_biometric_db"
COLLECTION_NAME = "users"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
user_collection = db[COLLECTION_NAME]
# In-memory database of users (now redundant, but kept for reference, will be removed in the future)
user_database: Dict[str, Dict] = {}
# Mapping of session tokens to user IDs
active_sessions: Dict[str, str] = {}
# Define data models
class UserCreate(BaseModel):
    username: str
    full_name: str
class User(BaseModel):
    user_id: str
    username: str
    full_name: str
class EnrollmentResponse(BaseModel):
    success: bool
    message: str
    user_id: str
class AuthenticationResponse(BaseModel):
    authenticated: bool
    session_token: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    message: str
class Enrollment(BaseModel):
    user_id: str
    embedding: List[float]
# Define storage paths
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
# EMBEDDING_DIR = os.path.join(os.path.dirname(__file__), "embeddings") # No longer needed
# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(EMBEDDING_DIR, exist_ok=True) # No longer needed
def save_embedding(user_id: str, embeddings: List[np.ndarray], R1_list: List[np.ndarray], M_list: List[np.ndarray], full_name: str = ""):
    """Save user enrollment data to MongoDB"""
    data = {
        "user_id": user_id,
        "full_name": full_name,
        "embeddings": embeddings,
        "R1_list": R1_list,
        "M_list": M_list,
    }
    # Convert numpy arrays to lists for serialization
    data["embeddings"] = [embedding.tolist() for embedding in embeddings]
    data["R1_list"] = [r1.tolist() for r1 in R1_list]
    data["M_list"] = [m.tolist() for m in M_list]
    user_collection.update_one(
        {"user_id": user_id},
        {"$set": data},
        upsert=True  # Creates a new document if user_id doesn't exist
    )
def load_embedding(user_id: str):
    """Load user enrollment data from MongoDB"""
    user_data = user_collection.find_one({"user_id": user_id})
    if not user_data:
        raise FileNotFoundError(f"No enrollment data found for user {user_id}")
    # Convert lists back to numpy arrays
    user_data["embeddings"] = [np.array(embedding) for embedding in user_data["embeddings"]]
    user_data["R1_list"] = [np.array(r1) for r1 in user_data["R1_list"]]
    user_data["M_list"] = [np.array(m) for m in user_data["M_list"]]
    return user_data
def process_image(image_path: str) -> np.ndarray:
    """Process an image to extract facial embedding without OPD protection"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")
    
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = fd.process(rgb)
        if not results.detections:
            raise ValueError("No face detected")
        
        box = results.detections[0].location_data.relative_bounding_box
        h, w = image.shape[:2]
        x1 = max(0, int((box.xmin - 0.1) * w))
        y1 = max(0, int((box.ymin - 0.1) * h))
        x2 = min(w, int((box.xmin + box.width + 0.1) * w))
        y2 = min(h, int((box.ymin + box.height + 0.1) * h))
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            raise ValueError("Face crop failed")
        
        face = cv2.resize(face, (224, 224))
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = Tr(Image.fromarray(face_rgb)).unsqueeze(0)
    
    with torch.no_grad():
        embedding = resnet_50(tensor).squeeze().numpy()
    
    # Just normalize the embedding without applying OPD
    embedding = embedding / np.linalg.norm(embedding)
    return embedding

def verify_similarity(enrolled_data, verification_embedding: np.ndarray) -> tuple:
    """
    Compare a stored user template against a new verification embedding.
    1) Quick cosine check (fast path).
    2) Secure dot-product check (slow path) if cosine fails.
    Prints exactly one debug line per call.
    """

    # 1. Normalize & apply One-Parameter Defense to the incoming embedding
    verification_embedding = verification_embedding / np.linalg.norm(verification_embedding)
    verification_embedding = one_parameter_defense(verification_embedding, m=3, seed=1234)

    # 2. Retrieve stored (protected) embedding and encryption key
    enrolled_embedding = enrolled_data["embeddings"][0]  # Already OPD-protected
    R1 = enrolled_data["R1_list"][0]
    M_enc = enrolled_data["M_list"][0]  # This is N_enc (encrypted noise)

    # === Fast Path: Cosine Similarity ===
    cosine_sim = float(np.dot(enrolled_embedding, verification_embedding))
    if cosine_sim > 0.32:  # Increase threshold for cosine similarity
        print(f"[VERIFY] Cosine-only pass: sim={cosine_sim:.4f}")
        return True, cosine_sim

    # === Slow Path: Secure Dot-Product Matching ===

    # Convert both embeddings to Q1.8 fixed-point
    X = float_to_q1_8(enrolled_embedding)
    Y = float_to_q1_8(verification_embedding)

    # Generate fresh random noise N₂ and encrypt it with a fresh pad
    N2 = float_to_q1_8(np.random.uniform(-1, 1, size=Y.shape[0]) * 50)
    Y_enc, N2_enc, _ = secure_enrollment(Y, N2)

    # Decrypt stored enrollment share (X_enc, M_enc) using R1
    X_enc = np.bitwise_xor(X, R1)

    # Now distribute shares to the two “parties” in this simulated protocol
    P1, P2 = distribute_template_shares(X_enc, Y_enc, M_enc, N2_enc)

    # --- Octet Generation & Secure AND (masked dot-product) ---
    def int_to_bits(arr, bit_width=16):
        bits = np.unpackbits(arr.view(np.uint8))
        # reshape so each row is one element’s bits, then reverse to MSB last
        return bits.reshape(-1, bit_width)[:, ::-1]

    num_bits = 128
    num_elems = num_bits // 16

    octets_X = generate_octets(int_to_bits(P1['X'][:num_elems]).flatten()[:num_bits])
    octets_Y = generate_octets(int_to_bits(P2['Y'][:num_elems]).flatten()[:num_bits])
    octets_M = generate_octets(int_to_bits(P1['M'][:num_elems]).flatten()[:num_bits])
    octets_N = generate_octets(int_to_bits(P2['N'][:num_elems]).flatten()[:num_bits])

    raw_score = secure_and_masked_dot_product(P1, P2, octets_X, octets_Y, octets_M, octets_N)
    _, observed_hd = compute_hamming_distance_bits(octets_X, octets_Y)
    corrected_score = apply_correction(raw_score, observed_hd, expected_hd=0.25)

    # === Single Debug Print for Secure Path ===
    print(f"[VERIFY] Secure match: score={corrected_score:.2f}, HD={observed_hd:.2f}")

    # Final decision: allow a slightly lower corrected score if HD is low
    if corrected_score >= 1.5 or (observed_hd <= 0.40 and cosine_sim > 0.30):
        return True, corrected_score

    return False, corrected_score

def cleanup_temp_files(file_path: str):
    """Delete temporary files after processing"""
    if os.path.exists(file_path):
        os.remove(file_path)
def enroll_with_multiple_images(user_id, image_paths, full_name=None):
    """Enroll user with multiple face images, applying OPD only once after averaging"""
    embeddings = []
    
    # Process 3 images and get raw embeddings
    for path in image_paths:
        embedding = process_image(path)
        embeddings.append(embedding)
    
    # Average the embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding /= np.linalg.norm(avg_embedding)
    
    # Apply OPD only once to the averaged embedding
    protected_embedding = one_parameter_defense(avg_embedding, m=2, seed=1234)
    
    # Generate noise vector for masking
    noise_vector = np.random.uniform(-1, 1, size=protected_embedding.shape[0]) * 50
    
    # Q1.8 fixed-point conversion
    X = float_to_q1_8(protected_embedding)
    N = float_to_q1_8(noise_vector)
    
    # Encrypt both using the same R
    X_enc, N_enc, R1 = secure_enrollment(X, N)
    
    # Save enrollment data
    save_embedding(user_id, [protected_embedding], [R1], [N_enc], full_name)
    
    # Update enrollment status in database
    user_collection.update_one({"user_id": user_id}, {"$set": {"enrolled": True}}, upsert=True)
    
    return EnrollmentResponse(
        success=True,
        message="User enrolled successfully with 3-image averaged embedding and noise",
        user_id=user_id
    )

@app.post("/users/", response_model=User)
async def create_user(user: UserCreate):
    """Create a new user in the system"""
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "username": user.username,
        "full_name": user.full_name,
        "enrolled": False
    }
    # user_database[user_id] = user_data # Remove as MongoDB is the new source of truth
    user_collection.insert_one(user_data)  # Store user in MongoDB
    return user_data
@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: str):
    """Get user information"""
    # if user_id not in user_database: # Now get from MongoDB
    # raise HTTPException(status_code=404, detail="User not found")
    user_data = user_collection.find_one({"user_id": user_id})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    return User(**user_data)

@app.post("/enroll_multiple/", response_model=EnrollmentResponse)
async def enroll_user_multiple(
    background_tasks: BackgroundTasks,
    full_name: str = Form(...),
    face_images: List[str] = Form(...)
):
    """Enroll user with multiple face images (in base64 format)"""
    if len(face_images) < 3:
        raise HTTPException(status_code=400, detail="At least 3 face images are required")

    user_id = str(uuid.uuid4())

    # Create user in DB
    user_collection.insert_one({
        "user_id": user_id,
        "full_name": full_name,
        "enrolled": False,
    })

    # Save uploaded images to disk
    image_paths = []
    for i, image_data in enumerate(face_images):
        image_path = os.path.join(UPLOAD_DIR, f"{user_id}_enrollment_{i}_{uuid.uuid4()}.jpg")
        try:
            # Handle both formats: with data:image/jpeg;base64 prefix or without
            if ',' in image_data:
                image_data = image_data.split(',')[1]  # Remove base64 prefix if present
            
            # Decode base64 string to image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            cv2.imwrite(image_path, img)
            image_paths.append(image_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not decode image {i}: {str(e)}")

    try:
        # Enroll user with images
        response = enroll_with_multiple_images(user_id, image_paths, full_name)

        # Mark user enrolled
        user_collection.update_one({"user_id": user_id}, {"$set": {"enrolled": True}})

        # Cleanup temp files
        for path in image_paths:
            background_tasks.add_task(cleanup_temp_files, path)

        return response

    except Exception as e:
        for path in image_paths:
            background_tasks.add_task(cleanup_temp_files, path)
        raise HTTPException(status_code=500, detail=f"Enrollment failed: {str(e)}")
@app.post("/authenticate/", response_model=AuthenticationResponse)
async def authenticate_user(background_tasks: BackgroundTasks, face_image: str = Form(...)):
    """Authenticate a user using facial biometric with live capture."""
    try:
        # Handle both formats: with data:image/jpeg;base64 prefix or without
        if ',' in face_image:
            face_image = face_image.split(',')[1]  # Remove data:image/jpeg;base64, prefix
        
        # Decode base64 data
        image_data = base64.b64decode(face_image)
        # Convert base64 data to cv2 image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
        
        # Save the image to a temporary file for processing
        image_path = os.path.join(UPLOAD_DIR, f"auth_{uuid.uuid4()}.jpg")
        cv2.imwrite(image_path, img)
        
        # Process image to extract verification embedding (raw)
        raw_embedding = process_image(image_path)
        
        # Apply OPD to raw embedding for verification
        verification_embedding = one_parameter_defense(raw_embedding, m=3, seed=1234)
        
        # Try to match against all enrolled users
        best_match = None
        best_score = 0
        user_name = None
        
        # Define minimum acceptable score threshold
        MIN_ACCEPTABLE_SCORE = 2.0  # Adjust based on experiments
        
        # Track all strong matches
        strong_matches = []
        
        for user_data in user_collection.find({"enrolled": True}):
            user_id = user_data["user_id"]
            try:
                # Load user's enrolled data
                enrolled_data = load_embedding(user_id)
                
                # Compare embeddings with user_id for logging
                is_authenticated, score = verify_similarity(enrolled_data, verification_embedding)
                
                # Track all scores for debugging
                print(f"User {user_id} ({user_data.get('full_name', 'Unknown')}) authentication result: {is_authenticated}, score: {score:.2f}")
                
                if is_authenticated:
                    # Track strong matches for potential conflicts
                    if score > MIN_ACCEPTABLE_SCORE:
                        strong_matches.append((user_id, score, user_data.get("full_name", "Unknown")))
                    
                    # Update best match if this is better than previous
                    if best_match is None or score > best_score:
                        best_match = user_id
                        best_score = score
                        user_name = user_data.get("full_name", "Unknown")
            except FileNotFoundError:
                # Skip users with missing embeddings
                continue
            except Exception as e:
                print(f"Error verifying user {user_id}: {str(e)}")
                continue
        
        # Log warning if multiple strong matches found
        if len(strong_matches) > 1:
            print(f"WARNING: Multiple strong matches found: {strong_matches}")
        
        # Reject match if score is too low
        if best_match and best_score < MIN_ACCEPTABLE_SCORE:
            print(f"Match found but score too low: {best_score}")
            best_match = None  # Reject the match
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_files, image_path)
        
        if best_match:
            # Generate session token for authenticated user
            session_token = str(uuid.uuid4())
            active_sessions[session_token] = best_match
            return AuthenticationResponse(
                authenticated=True,
                session_token=session_token,
                user_id=best_match,
                user_name=user_name,
                message=f"Authentication successful with score: {best_score:.2f}"
            )
        else:
            return AuthenticationResponse(
                authenticated=False,
                session_token=None,
                user_id=None,
                user_name=None,
                message="Authentication failed: No matching user found or score too low"
            )
    except Exception as e:
        # Schedule cleanup even on error
        if 'image_path' in locals():
            background_tasks.add_task(cleanup_temp_files, image_path)
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")
@app.post("/logout/")
async def logout(session_token: str):
    """Logout a user by invalidating their session token"""
    if session_token in active_sessions:
        del active_sessions[session_token]
        return {"success": True, "message": "Logged out successfully"}
    else:
        raise HTTPException(status_code=400, detail="Invalid session token")
@app.get("/healthcheck")
async def healthcheck():
    """API health check endpoint"""
    return {"status": "online", "service": "Face Biometric Authentication API"}
@app.post("/debug_enrollment/{user_id}")
async def debug_enrollment(user_id: str):
    """Debug endpoint to check enrollment data"""
    # if user_id not in user_database or not user_database[user_id].get("enrolled", False): # Now get from MongoDB
    # raise HTTPException(status_code=404, detail="User not enrolled")
    user_data = user_collection.find_one({"user_id": user_id, "enrolled": True})
    if not user_data:
        raise HTTPException(status_code=404, detail="User not enrolled")
    try:
        # Load enrollment data
        enrolled_data = load_embedding(user_id)
        # Return summary of data shapes for debugging
        return {
            "embedding_shape": enrolled_data["embeddings"][0].shape,
            "R1_shape": enrolled_data["R1_list"][0].shape,
            "M_shape": enrolled_data["M_list"][0].shape,
            "embedding_stats": {
                "mean": float(np.mean(enrolled_data["embeddings"][0])),
                "std": float(np.std(enrolled_data["embeddings"][0])),
                "min": float(np.min(enrolled_data["embeddings"][0])),
                "max": float(np.max(enrolled_data["embeddings"][0]))
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading enrollment data: {str(e)}")
def restore_user_database():
    """This function is not needed anymore, as MongoDB is the source of truth now."""
    pass  # Remove this function entirely. The database is not "restored" from pickle files, but is authoritative.
if __name__ == "__main__":
    # restore_user_database() # Remove this line. It's no longer needed.
    uvicorn.run(app, host="0.0.0.0", port=8000)
