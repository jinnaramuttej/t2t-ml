import os
import json
import csv
import time
import cv2
from PIL import Image
from unittest.mock import patch
import requests

from server import score_frames, decide, run_vlm_check, ClipScorer
import server

# Ensure scorer is initialized
if not hasattr(server, 'scorer') or server.scorer is None:
    server.scorer = ClipScorer()

def evaluate_baseline():
    print("\n--- Evaluating Baseline (22 Submissions) ---")
    labels = {}
    with open("audit_labels.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['folder']] = row
            
    folders = sorted([f for f in os.listdir("submissions") if os.path.isdir(os.path.join("submissions", f))])
    
    results = []
    
    for folder in folders:
        folder_path = os.path.join("submissions", folder)
        meta_path = os.path.join(folder_path, "meta.json")
        f1_path = os.path.join(folder_path, "frame1.jpg")
        f2_path = os.path.join(folder_path, "frame2.jpg")
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        img1 = Image.open(f1_path).convert('RGB')
        img2 = Image.open(f2_path).convert('RGB')
        tap = meta.get("tap_point", {"x": 0.5, "y": 0.5})
        
        t0 = time.time()
        # Non-VLM pipeline
        A, B, C, B_full1, B_full2, item_still_in_hand = score_frames(img1, img2, tap["x"], tap["y"])
        verdict, reason = decide(A, B, C)
        
        # VLM
        vlm_verdict = "N/A"
        reason_code = "N/A"
        if verdict == "approve":
            with open(f1_path, "rb") as f1, open(f2_path, "rb") as f2:
                b1, b2 = f1.read(), f2.read()
            vlm_res = run_vlm_check(None, b1, b2)
            vlm_verdict = vlm_res.get("vlm_verdict", "UNCERTAIN")
            reason_code = vlm_res.get("reason_code", "N/A")
            if vlm_verdict == "MISSED":
                verdict = "reject"
                reason = "missed bin (VLM)"
            elif vlm_verdict == "UNCERTAIN":
                verdict = "manual_review"
                reason = "uncertain disposal (VLM)"
                
        latency = time.time() - t0
        ground_truth = labels.get(folder, {}).get("valid_disposal", "")
        # For our baseline, empty means it wasn't a valid disposal according to audit report, unless we infer it
        
        results.append({
            "id": folder[:15],
            "truth": ground_truth,
            "vlm": vlm_verdict,
            "vlm_reason": reason_code,
            "final": verdict,
            "latency": latency
        })
        print(f"[{folder[:15]}] Final: {verdict:15s} VLM: {vlm_verdict:10s} Latency: {latency:.1f}s")
        
    return results


def evaluate_adversarial_clips():
    print("\n--- Evaluating New Adversarial Test Set (Clips) ---")
    clip_files = sorted([f for f in os.listdir("clips") if f.endswith(".mp4")])
    
    for c in clip_files:
        path = os.path.join("clips", c)
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        if not frames: continue
        
        img1 = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
        img2 = Image.fromarray(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
        
        with open(path, "rb") as f:
            v_bytes = f.read()
            
        t0 = time.time()
        A, B, C, B_full1, B_full2, item_still_in_hand = score_frames(img1, img2, 0.5, 0.5)
        verdict, reason = decide(A, B, C)
        
        vlm_verdict = "N/A"
        reason_code = "N/A"
        
        # We always run VLM for evaluation reporting, even if rejected by CLIP
        vlm_res = run_vlm_check(v_bytes, b"", b"")
        vlm_verdict = vlm_res.get("vlm_verdict", "UNCERTAIN")
        reason_code = vlm_res.get("reason_code", "N/A")
        
        if verdict == "approve":
            if vlm_verdict == "MISSED":
                verdict = "reject"
            elif vlm_verdict == "UNCERTAIN":
                verdict = "manual_review"
                
        latency = time.time() - t0
        print(f"[{c}] Final: {verdict:15s} VLM: {vlm_verdict:10s} (Reason: {reason_code}) Latency: {latency:.1f}s")


def test_api_failures():
    print("\n--- Testing API Failures and Malformed Responses ---")
    
    with open("submissions/20261008_140713_3200f3e0-9c4c-4279-b776-c2d1f5662ade/frame1.jpg", "rb") as f1, open("submissions/20261008_140713_3200f3e0-9c4c-4279-b776-c2d1f5662ade/frame2.jpg", "rb") as f2:
        b1, b2 = f1.read(), f2.read()
        
    def mock_post_timeout(*args, **kwargs):
        raise requests.exceptions.Timeout("Connection timed out")
        
    def mock_post_malformed(*args, **kwargs):
        class MockResponse:
            def json(self): return {"choices": [{"message": {"content": "This is just text, not JSON!"}}]}
            def raise_for_status(self): pass
        return MockResponse()
        
    with patch("requests.post", side_effect=mock_post_timeout):
        res1 = run_vlm_check(None, b1, b2)
        print(f"Timeout Test -> vlm_verdict: {res1['vlm_verdict']}, Note: {res1['note'][:25]}...")
        
    with patch("requests.post", side_effect=mock_post_malformed):
        res2 = run_vlm_check(None, b1, b2)
        print(f"Malformed JSON Test -> vlm_verdict: {res2['vlm_verdict']}")


if __name__ == "__main__":
    test_api_failures()
    evaluate_baseline()
    evaluate_adversarial_clips()
