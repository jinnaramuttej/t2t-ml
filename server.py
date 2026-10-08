# Requirements: fastapi, uvicorn, python-multipart

import os
import time
import json
import uuid
import datetime
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

import imagehash
from t2t_validator import ClipScorer, decide, is_duplicate

app = FastAPI()

# Config from environment
T2T_TOKEN = os.getenv("T2T_TOKEN", "")
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")

if ALLOWED_ORIGINS:
    origins = [orig.strip() for orig in ALLOWED_ORIGINS.split(",") if orig.strip()]
else:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory sessions
sessions = {}

# Global scorer
scorer = None

@app.on_event("startup")
def startup_event():
    global scorer
    print("Loading ClipScorer...")
    scorer = ClipScorer()
    os.makedirs("submissions", exist_ok=True)
    if not os.path.exists("submissions/hashes.json"):
        with open("submissions/hashes.json", "w") as f:
            json.dump([], f)

@app.middleware("http")
async def verify_token(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)
    
    # We apply token validation to all routes as requested
    if request.url.path not in ["/docs", "/openapi.json", "/redoc"]:
        token = request.headers.get("x-api-token")
        if not token or token != T2T_TOKEN:
            return Response(content="Unauthorized", status_code=401)
            
    return await call_next(request)

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    now = time.time()
    sessions[session_id] = {
        "created_at": now,
        "used": False
    }
    return {"session_id": session_id, "expires_in": 300}

def get_crop(image: Image.Image, tap_x: float, tap_y: float) -> Image.Image:
    w, h = image.size
    crop_w = w * 0.5
    crop_h = h * 0.5
    
    center_x = tap_x * w
    center_y = tap_y * h
    
    left = center_x - crop_w / 2
    top = center_y - crop_h / 2
    right = center_x + crop_w / 2
    bottom = center_y + crop_h / 2
    
    # Clamp to image bounds
    if left < 0:
        right += abs(left)
        left = 0
    if top < 0:
        bottom += abs(top)
        top = 0
    if right > w:
        left -= (right - w)
        right = w
    if bottom > h:
        top -= (bottom - h)
        bottom = h
        
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    
    return image.crop((left, top, right, bottom))

@app.post("/validate")
async def validate(
    request: Request,
    session_id: str = Form(...),
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    tap_x: Optional[float] = Form(None),
    tap_y: Optional[float] = Form(None),
    manual: bool = Form(False),
    test_type: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    x_debug_token: Optional[str] = Header(None)
):
    # Session validation
    now = time.time()
    if session_id not in sessions:
        raise HTTPException(status_code=400, detail="Unknown or missing session ID")
    
    session = sessions[session_id]
    if session["used"]:
        raise HTTPException(status_code=400, detail="Session already used")
        
    age = now - session["created_at"]
    if age > 300:
        raise HTTPException(status_code=400, detail="Session expired")
    if age < 4:
        raise HTTPException(status_code=400, detail="Session created less than 4 seconds ago")
        
    session["used"] = True
    
    # File validation
    f1_bytes = await frame1.read()
    f2_bytes = await frame2.read()
    
    if len(f1_bytes) > 1.5 * 1024 * 1024 or len(f2_bytes) > 1.5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (>1.5MB)")
        
    try:
        img1 = Image.open(BytesIO(f1_bytes))
        if img1.format not in ["JPEG", "MPO"]:
            raise ValueError()
        img1.load()
        img1 = img1.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="frame1 is not a valid JPEG")
        
    try:
        img2 = Image.open(BytesIO(f2_bytes))
        if img2.format not in ["JPEG", "MPO"]:
            raise ValueError()
        img2.load()
        img2 = img2.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="frame2 is not a valid JPEG")

    # If device was missing from Form, try to fetch from User-Agent
    if not device:
        device = request.headers.get("user-agent", "")

    # Scoring
    A = scorer.has_object(img1)
    
    if tap_x is not None and tap_y is not None:
        img1_crop = get_crop(img1, tap_x, tap_y)
        B = scorer.bin_present(img1_crop)
    else:
        B = scorer.bin_present(img1)
        
    C = scorer.hand_empty(img2)
    
    # Log only
    B_full1 = scorer.bin_present(img1)
    B_full2 = scorer.bin_present(img2)
    item_still_in_hand = scorer.has_object(img2)
    
    # Duplicates handling
    hashes_path = "submissions/hashes.json"
    try:
        with open(hashes_path, "r") as f:
            persisted_hashes_hex = json.load(f)
        persisted_hashes = [imagehash.hex_to_hash(h) for h in persisted_hashes_hex]
    except Exception:
        persisted_hashes_hex = []
        persisted_hashes = []
        
    is_dup = is_duplicate(img1, persisted_hashes)
    
    new_hash = imagehash.phash(img1)
    persisted_hashes_hex.append(str(new_hash))
    with open(hashes_path, "w") as f:
        json.dump(persisted_hashes_hex, f)
        
    # Decision logic
    verdict, reason = decide(A, B, C, is_dup=is_dup)
    
    if manual and verdict == "approve":
        verdict = "manual_review"
        reason = "manual review requested"
        
    # User-facing message
    message = ""
    if verdict == "approve":
        message = "Success"
    elif verdict == "manual_review":
        message = "sent for review"
    elif verdict == "reject":
        if "duplicate" in reason:
            message = "duplicate submission"
        elif "has_object" in reason:
            message = "no item seen in hand"
        elif "bin_present" in reason:
            message = "bin not visible"
        elif "hand_empty" in reason:
            message = "open hand not confirmed"
        else:
            message = "submission rejected"
            
    # Storage
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = f"submissions/{timestamp}_{session_id}"
    os.makedirs(sub_dir, exist_ok=True)
    img1.save(os.path.join(sub_dir, "frame1.jpg"))
    img2.save(os.path.join(sub_dir, "frame2.jpg"))
    
    meta = {
        "scores": {
            "A": A,
            "B": B,
            "C": C,
            "B_full1": B_full1,
            "B_full2": B_full2,
            "item_still_in_hand": item_still_in_hand
        },
        "verdict": verdict,
        "test_type": test_type,
        "device": device,
        "lat": lat,
        "lng": lng,
        "manual": manual,
        "tap_point": {"x": tap_x, "y": tap_y}
    }
    with open(os.path.join(sub_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
        
    # Response
    response = {
        "verdict": verdict,
        "message": message
    }
    
    if DEBUG_TOKEN and x_debug_token == DEBUG_TOKEN:
        response["debug"] = {
            "A": A,
            "B": B,
            "B_full1": B_full1,
            "B_full2": B_full2,
            "C": C,
            "item_still_in_hand": item_still_in_hand,
            "is_dup": is_dup,
            "reason": reason
        }
        
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
