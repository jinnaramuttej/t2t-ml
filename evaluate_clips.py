"""
evaluate_clips.py
Evaluates the OpenCLIP T2T validator against a dataset of labeled clips.
"""

import os
import csv
import argparse
import sys
from pathlib import Path
from PIL import Image

try:
    import cv2
except ImportError:
    print("Please install opencv-python: pip install opencv-python")
    sys.exit(1)

import imagehash
from t2t_validator import ClipScorer, decide, is_duplicate

def extract_frames(video_path, t1, t2_from_end):
    """
    Extracts two frames from the video at t1 seconds and (duration - t2_from_end) seconds.
    Returns (img1_pil, img2_pil).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return None, None
        
    duration = frame_count / fps
    
    t1_actual = min(max(t1, 0.0), duration)
    t2_actual = min(max(duration - t2_from_end, 0.0), duration)
    
    # Extract f1
    cap.set(cv2.CAP_PROP_POS_MSEC, t1_actual * 1000)
    ret1, frame1 = cap.read()
    
    # Extract f2
    cap.set(cv2.CAP_PROP_POS_MSEC, t2_actual * 1000)
    ret2, frame2 = cap.read()
    
    cap.release()
    
    if not ret1 or not ret2:
        return None, None
        
    frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame2_rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    img1 = Image.fromarray(frame1_rgb)
    img2 = Image.fromarray(frame2_rgb)
    
    return img1, img2

def generate_contact_sheet(video_path, clip_name, output_dir):
    """
    Extracts one frame every 0.5s and saves to frames/contact/{clip}/
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        return
        
    duration = frame_count / fps
    clip_out_dir = Path(output_dir) / clip_name
    clip_out_dir.mkdir(parents=True, exist_ok=True)
    
    t = 0.0
    idx = 0
    while t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if not ret:
            break
        out_path = clip_out_dir / f"frame_{idx:04d}_{t:.1f}s.jpg"
        cv2.imwrite(str(out_path), frame)
        t += 0.5
        idx += 1
        
    cap.release()

def load_labels(csv_path):
    labels = []
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row)
    return labels

def main():
    parser = argparse.ArgumentParser(description="Evaluate T2T Validator.")
    parser.add_argument("--labels", default="clips/labels.csv", help="Path to labels.csv")
    parser.add_argument("--clips-dir", default="clips", help="Directory with clips")
    parser.add_argument("--frames-dir", default="frames", help="Directory to save extracted frames")
    parser.add_argument("--t1", type=float, default=1.0, help="Time for frame 1 in seconds")
    parser.add_argument("--t2-from-end", type=float, default=1.0, help="Time from end for frame 2 in seconds")
    parser.add_argument("--contact-sheet", action="store_true", help="Only generate contact sheets (1 frame / 0.5s)")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep after evaluation")
    args = parser.parse_args()
    
    labels = load_labels(args.labels)
    if not labels:
        print(f"No labels found at {args.labels}")
        sys.exit(1)
        
    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    clips_dir = Path(args.clips_dir)
    
    if args.contact_sheet:
        print("Generating contact sheets...")
        for row in labels:
            clip = row["clip"]
            clip_path = None
            
            # Resolve clip path
            if (clips_dir / clip).exists():
                clip_path = clips_dir / clip
            else:
                for ext in [".mp4", ".mov", ".MP4", ".MOV"]:
                    if (clips_dir / f"{clip}{ext}").exists():
                        clip_path = clips_dir / f"{clip}{ext}"
                        break
                        
            if clip_path and clip_path.exists():
                generate_contact_sheet(str(clip_path), clip, frames_dir / "contact")
            else:
                print(f"Clip not found: {clip}")
        print("Contact sheets generated. Exiting.")
        sys.exit(0)

    # Standard evaluation
    print("Loading model...")
    scorer = ClipScorer()
    seen_hashes = []
    
    results = []
    
    print("Evaluating clips...")
    for row in labels:
        clip = row["clip"]
        
        # Find clip file
        clip_path = None
        if (clips_dir / clip).exists():
            clip_path = clips_dir / clip
        else:
            for ext in [".mp4", ".mov", ".MP4", ".MOV"]:
                if (clips_dir / f"{clip}{ext}").exists():
                    clip_path = clips_dir / f"{clip}{ext}"
                    break
                    
        if not clip_path or not clip_path.exists():
            print(f"Warning: Clip {clip} not found or unreadable. Skipping.")
            continue
            
        # Extract frames
        img1, img2 = extract_frames(str(clip_path), args.t1, args.t2_from_end)
        if not img1 or not img2:
            print(f"Warning: Could not extract frames for {clip}. Skipping.")
            continue
            
        # Save frames
        base_clip_name = os.path.splitext(clip)[0]
        f1_path = frames_dir / f"{base_clip_name}_f1.jpg"
        f2_path = frames_dir / f"{base_clip_name}_f2.jpg"
        img1.save(f1_path)
        img2.save(f2_path)
        
        # Score
        A = scorer.has_object(img1)
        B = scorer.bin_present(img2)
        C = scorer.hand_empty(img2)
        
        if getattr(scorer, 'last_hand_crop', None) is not None:
            scorer.last_hand_crop.save(frames_dir / f"{base_clip_name}_f2_crop.jpg")
        else:
            print(f"no hand detected for {clip}")
        
        # Duplication
        h1 = imagehash.phash(img1)
        is_dup = is_duplicate(img1, seen_hashes)
        
        # Add to seen_hashes for future clips
        seen_hashes.append(h1)
        
        # Verdict
        verdict, reason = decide(A, B, C, is_dup=is_dup, gps_ok=True)
        
        results.append({
            "clip": clip,
            "A": A,
            "B": B,
            "C": C,
            "verdict": verdict,
            "reason": reason,
            "correct_decision": row["correct_decision"],
            "true_has_object": row["has_object_f1"],
            "true_bin_present": row["bin_present_f2"],
            "true_hand_empty": row["hand_empty_f2"],
            "is_dup": is_dup
        })
        
    # Save eval_results.csv
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "A", "B", "C", "verdict", "reason", "correct_decision"])
        writer.writeheader()
        for res in results:
            writer.writerow({
                "clip": res["clip"],
                "A": f'{res["A"]:.4f}',
                "B": f'{res["B"]:.4f}',
                "C": f'{res["C"]:.4f}',
                "verdict": res["verdict"],
                "reason": res["reason"],
                "correct_decision": res["correct_decision"]
            })
            
    # Print metrics
    if not results:
        print("No valid results to analyze.")
        sys.exit(0)
        
    total = len(results)
    correct_verdicts = sum(1 for r in results if r["verdict"] == r["correct_decision"])
    manual_reviews = sum(1 for r in results if r["verdict"] == "manual_review")
    
    print("\n" + "="*40)
    print(" FALSE APPROVALS (Critical Failures)")
    print("="*40)
    false_approvals = [r for r in results if r["verdict"] == "approve" and r["correct_decision"] != "approve"]
    if false_approvals:
        for r in false_approvals:
            print(f"- {r['clip']}: True decision = {r['correct_decision']}, A={r['A']:.2f}, B={r['B']:.2f}, C={r['C']:.2f}")
    else:
        print("None! Great job.")
        
    print("\n" + "="*40)
    print(" OVERALL METRICS")
    print("="*40)
    print(f"Total evaluated:     {total}")
    print(f"Overall accuracy:    {correct_verdicts/total*100:.1f}%")
    print(f"Manual review rate:  {manual_reviews/total*100:.1f}%")
    
    print("\n" + "="*40)
    print(" CONFUSION TABLE (Correct vs Verdict)")
    print("="*40)
    col_name = "True \\ Pred"
    print(f"{col_name:<15} | {'approve':<10} | {'reject':<10} | {'manual_review':<13}")
    print("-" * 55)
    for true_lbl in ["approve", "reject", "manual_review"]:
        preds = {"approve": 0, "reject": 0, "manual_review": 0}
        for r in results:
            if r["correct_decision"] == true_lbl:
                preds[r["verdict"]] = preds.get(r["verdict"], 0) + 1
        print(f"{true_lbl:<15} | {preds['approve']:<10} | {preds['reject']:<10} | {preds['manual_review']:<13}")
        
    # Per-check accuracy (0.5 cutoff)
    def check_acc(score_key, true_key):
        correct = 0
        valid = 0
        for r in results:
            t = r[true_key].lower().strip()
            if t in ['yes', 'no']:
                valid += 1
                pred = "yes" if r[score_key] >= 0.5 else "no"
                if pred == t:
                    correct += 1
        return correct / valid if valid > 0 else 0
        
    print("\n" + "="*40)
    print(" PER-CHECK ACCURACY (0.5 cutoff)")
    print("="*40)
    print(f"has_object (A):  {check_acc('A', 'true_has_object')*100:.1f}%")
    print(f"bin_present (B): {check_acc('B', 'true_bin_present')*100:.1f}%")
    print(f"hand_empty (C):  {check_acc('C', 'true_hand_empty')*100:.1f}%")
    
    # Sweep
    if args.sweep:
        print("\n" + "="*40)
        print(" THRESHOLD SWEEP")
        print("="*40)
        print(f"{'High':<5} | {'Low':<5} | {'False Approves':<15} | {'Manual Review Rate':<20}")
        print("-" * 55)
        for h in [0.6, 0.7, 0.8, 0.9]:
            for l in [0.2, 0.3, 0.4]:
                fa = 0
                mr = 0
                for r in results:
                    v, _ = decide(r["A"], r["B"], r["C"], is_dup=r["is_dup"], gps_ok=True, high=h, low=l)
                    if v == "approve" and r["correct_decision"] != "approve":
                        fa += 1
                    if v == "manual_review":
                        mr += 1
                mr_rate = mr / total * 100
                print(f"{h:<5.2f} | {l:<5.2f} | {fa:<15} | {mr_rate:<5.1f}%")

if __name__ == "__main__":
    main()
