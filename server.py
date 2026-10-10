# Requirements: fastapi, uvicorn, python-multipart

import os
import time
import json
import uuid
import datetime
import base64
import requests
from io import BytesIO
from typing import Optional

PROMPT_VERSION = "1.0"

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from PIL import Image

import imagehash
from t2t_validator import ClipScorer, decide, is_duplicate

app = FastAPI()

# Config from environment
T2T_TOKEN = os.getenv("T2T_TOKEN", "")
DEBUG_TOKEN = os.getenv("DEBUG_TOKEN", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "")
DEDUP = os.getenv("DEDUP", "1")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"REJECTED: Unhandled exception - {str(exc)}")
    return JSONResponse(status_code=500, content={"message": "Internal server error"})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    print(f"REJECTED: {exc.detail}")
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"REJECTED: Validation error - {str(exc)}")
    return JSONResponse(status_code=422, content={"message": "Invalid request parameters"})

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

def score_frames(img1, img2, tap_x, tap_y):
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
    
    return A, B, C, B_full1, B_full2, item_still_in_hand

def run_vlm_check(video_bytes: Optional[bytes], img1_bytes: bytes, img2_bytes: bytes) -> dict:
    result = {
        "vlm_verdict": "UNCERTAIN",
        "reason_code": None,
        "note": None,
        "sampling_strategy": "fallback_2_frames",
        "sampled_frame_timestamps": []
    }
    
    if not GROQ_API_KEY:
        print("Warning: GROQ_API_KEY is not set.")
        result["vlm_verdict"] = "IN_BIN"
        return result
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    images_content = []
    
    if video_bytes:
        import tempfile
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_video_path = temp_video.name
            
        import numpy as np
        
        cap = cv2.VideoCapture(temp_video_path)
        raw_frames = []
        timestamps = []
        
        while True:
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            ret, frame = cap.read()
            if not ret:
                break
            if not timestamps or pos_msec > timestamps[-1]:
                timestamps.append(pos_msec)
            else:
                timestamps.append(timestamps[-1] + 33.3)
            raw_frames.append(frame)
            
        n = len(raw_frames)
        strategy = ""
        final_indices = []
        
        if n > 0:
            if n <= 3:
                strategy = "uniform_fallback"
                final_indices = list(range(n))
            else:
                diffs = [0]
                for i in range(1, n):
                    gray1 = cv2.cvtColor(cv2.resize(raw_frames[i-1], (128, 128)), cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(cv2.resize(raw_frames[i], (128, 128)), cv2.COLOR_BGR2GRAY)
                    diffs.append(np.sum(cv2.absdiff(gray1, gray2)))
                    
                peak_idx = int(np.argmax(diffs))
                candidate_indices = {
                    max(0, peak_idx - 1): "before_peak",
                    peak_idx: "peak_motion",
                    min(n - 1, peak_idx + 1): "after_peak"
                }
                
                final_indices = sorted(list(set([idx for idx in candidate_indices.keys() if 0 <= idx < n])))
                if len(final_indices) < 3:
                    final_indices = sorted(list(set([0, n // 2, n - 1])))
                    strategy = "uniform_fallback"
                else:
                    strategy = "motion_peak_3"
            
            result["sampling_strategy"] = strategy
            print(f"VLM Sampling: {strategy} | Indices: {final_indices}")
            for idx in final_indices:
                ts = timestamps[idx]
                result["sampled_frame_timestamps"].append(ts)
                print(f"  Frame {idx}: TS {ts:.1f}ms")
                frame = cv2.resize(raw_frames[idx], (512, 512))
                _, buffer = cv2.imencode('.jpg', frame)
                b64 = base64.b64encode(buffer).decode('utf-8')
                images_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64}"
                    }
                })
        cap.release()
        os.remove(temp_video_path)
        
    if not images_content:
        for fb in [img1_bytes, img2_bytes]:
            b64 = base64.b64encode(fb).decode('utf-8')
            images_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}"
                }
            })
            
    system_prompt = """You are the disposal verification component of Trash2Treasure (T2T).

You will receive up to 3 ordered frames extracted from a waste-disposal recording. Determine whether the waste item was successfully deposited inside the intended bin.

Return ONLY one valid JSON object matching this schema:

{
  "verdict": "IN_BIN | MISSED | UNCERTAIN",
  "reason_code": "ITEM_ENTERED_BIN | ITEM_FELL_OUTSIDE | ITEM_NOT_VISIBLE | BIN_NOT_VISIBLE | TRAJECTORY_AMBIGUOUS | OCCLUSION | INSUFFICIENT_EVIDENCE",
  "note": "A brief factual observation, maximum 12 words."
}

Decision rules:
- IN_BIN: Visual evidence clearly supports the item entering the bin.
- MISSED: Visual evidence clearly shows the item falling outside the bin.
- UNCERTAIN: The available frames do not establish either outcome.
- Never infer successful disposal solely because the hand is empty afterward or a bin is visible.
- Do not assume the item entered the bin merely because it disappears between frames.
- If the item, its trajectory, or its destination is obscured or missing from the evidence, use UNCERTAIN.
- Describe only what is visible. Do not invent events between frames.
- Use the most specific applicable reason code. For UNCERTAIN, use the reason that best explains the missing evidence.
- Keep the note factual, concise, and useful for debugging.
- Do not include markdown, additional keys, or text outside the JSON object."""

    payload = {
        "model": "qwen/qwen3.8-27b",
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": images_content
            }
        ]
    }
    
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"VLM response: {answer}")
        
        try:
            import json
            clean_answer = answer
            if clean_answer.startswith("```json"):
                clean_answer = clean_answer[7:-3].strip()
            elif clean_answer.startswith("```"):
                clean_answer = clean_answer[3:-3].strip()
            parsed = json.loads(clean_answer)
            result["vlm_verdict"] = parsed.get("verdict", "UNCERTAIN")
            result["reason_code"] = parsed.get("reason_code")
            result["note"] = parsed.get("note")
        except Exception:
            if "IN_BIN" in answer.upper(): result["vlm_verdict"] = "IN_BIN"
            elif "MISSED" in answer.upper(): result["vlm_verdict"] = "MISSED"
            else: result["vlm_verdict"] = "UNCERTAIN"
    except Exception as e:
        print(f"VLM check failed: {e}")
        result["vlm_verdict"] = "ERROR"
        result["note"] = str(e)
        
    return result

@app.post("/validate")
async def validate(
    request: Request,
    session_id: str = Form(...),
    frame1: UploadFile = File(...),
    frame2: UploadFile = File(...),
    video: Optional[UploadFile] = File(None),
    tap_x: Optional[float] = Form(None),
    tap_y: Optional[float] = Form(None),
    manual: bool = Form(False),
    test_type: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
    lat: Optional[float] = Form(None),
    lng: Optional[float] = Form(None),
    f1_change: Optional[str] = Form(None),
    duration_ms: Optional[int] = Form(None),
    retries: Optional[int] = Form(None),
    x_debug_token: Optional[str] = Header(None)
):
    # Session validation
    now = time.time()
    start_time = now
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
    v_bytes = None
    if video:
        v_bytes = await video.read()
    
    
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
    A, B, C, B_full1, B_full2, item_still_in_hand = score_frames(img1, img2, tap_x, tap_y)
    
    # Duplicates handling
    hashes_path = "submissions/hashes.json"
    is_dup = False
    
    try:
        with open(hashes_path, "r") as f:
            data = json.load(f)
            
        valid_records = []
        for item in data:
            if isinstance(item, dict) and now - item.get("time", 0) <= 600:
                valid_records.append(item)
                
        persisted_hashes = [imagehash.hex_to_hash(r["hash"]) for r in valid_records]
    except Exception:
        valid_records = []
        persisted_hashes = []
        
    if DEDUP != "0":
        is_dup = is_duplicate(img1, persisted_hashes)
        
    new_hash = str(imagehash.phash(img1))
    valid_records.append({"hash": new_hash, "time": now})
    with open(hashes_path, "w") as f:
        json.dump(valid_records, f)
        
    # Decision logic
    verdict, reason = decide(A, B, C, is_dup=is_dup)
    
    # Run VLM check if CLIP approved
    vlm_passed = True
    vlm_result = None
    if verdict == "approve":
        print("Running VLM check...")
        vlm_result = run_vlm_check(v_bytes, f1_bytes, f2_bytes)
        vlm_verdict = vlm_result.get("vlm_verdict", "UNCERTAIN")
        if vlm_verdict == "MISSED":
            vlm_passed = False
            verdict = "reject"
            reason = "missed bin (VLM)"
        elif vlm_verdict == "UNCERTAIN":
            vlm_passed = False
            verdict = "manual_review"
            reason = "uncertain disposal (VLM)"
        elif vlm_verdict == "ERROR":
            vlm_passed = False
            verdict = "manual_review"
            reason = "vlm processing error"
        elif vlm_verdict != "IN_BIN":
            vlm_passed = False
            verdict = "manual_review"
            reason = f"unexpected vlm verdict: {vlm_verdict}"
    
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
            
    # Logging
    elapsed = time.time() - start_time
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    f1_chg_log = f" f1_change={f1_change}" if f1_change else ""
    print(f"[{timestamp}] session={session_id} verdict={verdict} reason='{reason}' A={A:.4f} B={B:.4f} B_full1={B_full1:.4f} B_full2={B_full2:.4f} C={C:.4f} item_still_in_hand={item_still_in_hand} is_dup={is_dup}{f1_chg_log} elapsed={elapsed:.2f}s")
    
    # Storage
    sub_dir = f"submissions/{timestamp}_{session_id}"
    os.makedirs(sub_dir, exist_ok=True)
    img1.save(os.path.join(sub_dir, "frame1.jpg"))
    img2.save(os.path.join(sub_dir, "frame2.jpg"))
    
    meta = {
        "submission_id": f"{timestamp}_{session_id}",
        "dataset_version": "v1.1",
        "scores": {
            "A": A,
            "B": B,
            "C": C,
            "B_full1": B_full1,
            "B_full2": B_full2,
            "item_still_in_hand": item_still_in_hand
        },
        "verdict": verdict,
        "reason": reason,
        "manual_review_status": verdict == "manual_review",
        "human_verified_label": None,
        "test_type": test_type,
        "device": device,
        "lat": lat,
        "lng": lng,
        "manual": manual,
        "tap_point": {"x": tap_x, "y": tap_y},
        "f1_change": f1_change,
        "vlm_passed": vlm_passed,
        "duration_ms": duration_ms,
        "retries": retries
    }
    
    if vlm_result:
        meta.update({
            "model_id": "qwen/qwen3.8-27b",
            "prompt_version": PROMPT_VERSION,
            "sampling_strategy": vlm_result.get("sampling_strategy"),
            "sampled_frame_timestamps": vlm_result.get("sampled_frame_timestamps"),
            "vlm_verdict": vlm_result.get("vlm_verdict"),
            "reason_code": vlm_result.get("reason_code"),
            "vlm_note": vlm_result.get("note")
        })
        
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
