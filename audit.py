import os
import sys
import json
import csv
import math
import argparse
from datetime import datetime
from statistics import median
from PIL import Image, ImageDraw, ImageFont
import imagehash
import numpy as np

# Use functions and constants from the validator directly instead of server.py
# to avoid starting the server or triggering env/global dependencies.
from t2t_validator import ClipScorer, decide, HIGH, LOW, get_hand_landmarks, get_hand_box

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
        
    return image.crop((max(0, left), max(0, top), min(w, right), min(h, bottom)))

def classify_device(ua):
    if not ua:
        return "other"
    ua_lower = ua.lower()
    if "; wv)" in ua_lower or "instagram" in ua_lower or "fban" in ua_lower or "fbav" in ua_lower or "whatsapp" in ua_lower or "line" in ua_lower:
        return "in-app or webview"
    elif "chrome" in ua_lower and "safari" in ua_lower:
        return "chrome"
    return "other"

def parse_time(folder_name):
    try:
        ts_str = folder_name.split("_")[0:2]
        return datetime.strptime("_".join(ts_str), "%Y%m%d_%H%M%S")
    except:
        return None

def create_contact_sheet(rows_data, out_path, is_scored):
    if not rows_data: return
    
    THUMB_W = 240
    row_heights = []
    thumbs_info = []
    
    for row in rows_data:
        idx_str, f1_path, f2_path, tap_x, tap_y, text_info = row
        img1 = Image.open(f1_path).convert("RGB")
        img2 = Image.open(f2_path).convert("RGB")
        
        # Resize preserving aspect ratio
        w, h = img1.size
        th = int(h * (THUMB_W / w))
        img1 = img1.resize((THUMB_W, th))
        
        w2, h2 = img2.size
        th2 = int(h2 * (THUMB_W / w2))
        img2 = img2.resize((THUMB_W, th2))
        
        # Draw tap point on frame 1
        if tap_x is not None and tap_y is not None:
            draw = ImageDraw.Draw(img1)
            cx = tap_x * THUMB_W
            cy = tap_y * th
            r = 5
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), outline="red", width=2)
            
        row_h = max(th, th2, 40)
        row_heights.append(row_h)
        thumbs_info.append((img1, img2, text_info, idx_str))
        
    PADDING = 10
    total_h = sum(row_heights) + PADDING * (len(row_heights) + 1)
    
    idx_width = 40
    text_width = 320 if is_scored else 0
    total_w = idx_width + THUMB_W*2 + PADDING*4 + text_width
    
    sheet = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(sheet)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
        
    y_offset = PADDING
    for i, (img1, img2, text_info, idx_str) in enumerate(thumbs_info):
        # Draw index
        draw.text((PADDING, y_offset + row_heights[i]//2), idx_str, fill="black", font=font)
        
        # Draw img1
        x1 = PADDING*2 + idx_width
        sheet.paste(img1, (x1, y_offset))
        
        # Draw img2
        x2 = x1 + THUMB_W + PADDING
        sheet.paste(img2, (x2, y_offset))
        
        # Draw text
        if is_scored:
            x3 = x2 + THUMB_W + PADDING
            draw.text((x3, y_offset), text_info, fill="black", font=font)
            
        y_offset += row_heights[i] + PADDING
        
    sheet.save(out_path, "JPEG", quality=80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, help="Run in LABEL MODE with the provided CSV")
    args = parser.parse_args()

    if args.labels:
        run_label_mode(args.labels)
        return

    print("Discovering submissions...")
    base_dir = "submissions"
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found.")
        sys.exit(1)
        
    folders = []
    skipped = []
    for d in os.listdir(base_dir):
        dp = os.path.join(base_dir, d)
        if not os.path.isdir(dp): continue
        
        f1 = os.path.join(dp, "frame1.jpg")
        f2 = os.path.join(dp, "frame2.jpg")
        meta = os.path.join(dp, "meta.json")
        
        if not (os.path.exists(f1) and os.path.exists(f2) and os.path.exists(meta)):
            skipped.append((d, "Missing required files"))
            continue
            
        t = parse_time(d)
        if not t:
            skipped.append((d, "Invalid timestamp format"))
            continue
            
        folders.append((t, d, dp, f1, f2, meta))
        
    folders.sort(key=lambda x: x[0])
    
    if skipped:
        print("Skipped folders:")
        for sf, reason in skipped:
            print(f"  {sf}: {reason}")
            
    print(f"Found {len(folders)} valid submissions.")
    
    print("Loading ClipScorer...")
    scorer = ClipScorer()
    
    results = []
    seen_hashes = []
    
    os.makedirs("audit_sheets", exist_ok=True)
    
    sheet_rows_blind = []
    sheet_rows_scored = []
    sheet_idx = 1
    
    print("Rescoring...")
    for idx, (t, d, dp, f1, f2, meta_path) in enumerate(folders):
        idx_str = f"{idx+1:02d}"
        
        with open(meta_path, "r") as f:
            meta_data = json.load(f)
            
        tap_pt = meta_data.get("tap_point", {})
        tap_x = tap_pt.get("x")
        tap_y = tap_pt.get("y")
        test_type = meta_data.get("test_type", "")
        manual = meta_data.get("manual", False)
        device = meta_data.get("device", "")
        dev_class = classify_device(device)
        
        stored_scores = meta_data.get("scores", {})
        
        img1 = Image.open(f1).convert("RGB")
        img2 = Image.open(f2).convert("RGB")
        
        A = np.float32(scorer.has_object(img1))
        
        if tap_x is not None and tap_y is not None:
            img1_crop = get_crop(img1, tap_x, tap_y)
            B = np.float32(scorer.bin_present(img1_crop))
        else:
            B = np.float32(scorer.bin_present(img1))
            
        C = np.float32(scorer.hand_empty(img2))
        
        B_full1 = np.float32(scorer.bin_present(img1))
        B_full2 = np.float32(scorer.bin_present(img2))
        
        verdict, reason = decide(A, B, C, high=HIGH, low=LOW)
        weakest = reason.replace("weakest check: ", "").strip() if "weakest check" in reason else ""
        
        # Hand computations
        h1_box = get_hand_box(img1)
        hand_f1 = h1_box is not None
        
        lm2_info = get_hand_landmarks(img2)
        h2_box = get_hand_box(landmarks_info=lm2_info)
        hand_f2 = h2_box is not None
        
        hand_area_f2 = 0.0
        digits_f2 = 0
        if lm2_info:
            landmarks, w, h = lm2_info
            if h2_box:
                x_min, y_min, x_max, y_max = h2_box
                hand_area_f2 = ((x_max - x_min) * (y_max - y_min)) / (w * h)
                
            def dist(lm1, lm2):
                return math.hypot((lm1.x * w) - (lm2.x * w), (lm1.y * h) - (lm2.y * h))
                
            if dist(landmarks[0], landmarks[8]) > dist(landmarks[0], landmarks[6]): digits_f2 += 1
            if dist(landmarks[0], landmarks[12]) > dist(landmarks[0], landmarks[10]): digits_f2 += 1
            if dist(landmarks[0], landmarks[16]) > dist(landmarks[0], landmarks[14]): digits_f2 += 1
            if dist(landmarks[0], landmarks[20]) > dist(landmarks[0], landmarks[18]): digits_f2 += 1
            if dist(landmarks[17], landmarks[4]) > dist(landmarks[17], landmarks[3]): digits_f2 += 1
            
        # Hashes
        current_hash = imagehash.phash(img1)
        min_dist = None
        for h_prev in seen_hashes:
            d_val = current_hash - h_prev
            if min_dist is None or d_val < min_dist:
                min_dist = d_val
        seen_hashes.append(current_hash)
        
        res = {
            "index": idx_str, "folder": d, "time": t.strftime("%Y-%m-%d %H:%M:%S"),
            "test_type": test_type, "manual": manual, "device_class": dev_class,
            "stored_A": stored_scores.get("A",""), "stored_B": stored_scores.get("B",""), "stored_C": stored_scores.get("C",""),
            "new_A": float(A), "new_B": float(B), "new_C": float(C),
            "new_B_full1": float(B_full1), "new_B_full2": float(B_full2),
            "new_verdict": verdict, "weakest_check": weakest,
            "tap_x": tap_x if tap_x is not None else "", "tap_y": tap_y if tap_y is not None else "",
            "hand_f1": hand_f1, "hand_f2": hand_f2, "hand_area_f2": float(hand_area_f2),
            "digits_f2": digits_f2, "min_hash_dist": min_dist if min_dist is not None else ""
        }
        results.append(res)
        
        # Prepare sheet data
        text_info = f"A: {A:.4f}\nB: {B:.4f}\nC: {C:.4f}\nB_f1: {B_full1:.4f}\nB_f2: {B_full2:.4f}\n\nVerdict: {verdict}\nWeakest: {weakest}"
        row_data = (idx_str, f1, f2, tap_x, tap_y, text_info)
        sheet_rows_blind.append((idx_str, f1, f2, tap_x, tap_y, ""))
        sheet_rows_scored.append(row_data)
        
        if len(sheet_rows_blind) == 6 or idx == len(folders) - 1:
            create_contact_sheet(sheet_rows_blind, f"audit_sheets/blind_{sheet_idx:02d}.jpg", is_scored=False)
            create_contact_sheet(sheet_rows_scored, f"audit_sheets/scored_{sheet_idx:02d}.jpg", is_scored=True)
            sheet_rows_blind = []
            sheet_rows_scored = []
            sheet_idx += 1
            
    # Outputs
    with open("audit.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    with open("audit_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "folder", "f1_item_in_hand", "f2_bin_visible", "f2_open_palm_empty", "valid_disposal", "fake_type", "notes"])
        for r in results:
            writer.writerow([r["index"], r["folder"], "", "", "", "", "", ""])
            
    # Summary
    verdict_counts = {}
    weak_counts = {}
    dev_counts = {}
    hand_f2_count = 0
    manual_count = 0
    A_vals, B_vals, Bf1_vals, Bf2_vals, C_vals = [], [], [], [], []
    dup_pairs = []
    
    for i, r in enumerate(results):
        v = r["new_verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        if v == "reject":
            w = r["weakest_check"]
            weak_counts[w] = weak_counts.get(w, 0) + 1
            
        dc = r["device_class"]
        dev_counts[dc] = dev_counts.get(dc, 0) + 1
        
        if r["hand_f2"]: hand_f2_count += 1
        if r["manual"]: manual_count += 1
        
        A_vals.append(r["new_A"])
        B_vals.append(r["new_B"])
        Bf1_vals.append(B_full1)
        Bf2_vals.append(B_full2)
        C_vals.append(r["new_C"])
        
        if r["min_hash_dist"] != "" and r["min_hash_dist"] <= 5:
            dup_pairs.append(r["index"])
            
    gaps = []
    for i in range(1, len(folders)):
        gap = (folders[i][0] - folders[i-1][0]).total_seconds()
        gaps.append(gap)
        
    lines = []
    lines.append("=== AUDIT SUMMARY ===")
    lines.append(f"Total Submissions: {len(results)}")
    lines.append(f"Verdicts: {verdict_counts}")
    lines.append(f"Weakest checks among rejects: {weak_counts}")
    lines.append(f"Device breakdown: {dev_counts}")
    lines.append(f"Manual requests: {manual_count}")
    lines.append(f"Hand detected in frame 2: {hand_f2_count}/{len(results)} ({(hand_f2_count/max(1,len(results)))*100:.1f}%)")
    lines.append(f"Duplicates (dist <= 5): {len(dup_pairs)} submissions -> {dup_pairs}")
    if gaps:
        lines.append(f"Gaps (sec): min={min(gaps)}, median={median(gaps):.1f}, max={max(gaps)}")
        
    def get_stats(arr, name):
        if not arr: return f"{name}: N/A"
        return f"{name}: min={min(arr):.4f}, med={median(arr):.4f}, max={max(arr):.4f}"
        
    lines.append("Scores distributions:")
    lines.append(get_stats(A_vals, "A"))
    lines.append(get_stats(B_vals, "B"))
    lines.append(get_stats(Bf1_vals, "B_full1"))
    lines.append(get_stats(Bf2_vals, "B_full2"))
    lines.append(get_stats(C_vals, "C"))
    
    summ_text = "\n".join(lines)
    print("\n" + summ_text)
    with open("audit_summary.txt", "w") as f:
        f.write(summ_text)
        
    print("\nFiles generated:")
    print("  audit.csv")
    print("  audit_summary.txt")
    print("  audit_labels.csv (template)")
    print("  audit_sheets/ (blind and scored contact sheets)")
    
    print("\nExact commands:")
    print("  python audit.py")
    print("  python audit.py --labels audit_labels.csv")

def run_label_mode(labels_csv):
    if not os.path.exists(labels_csv):
        print(f"File {labels_csv} not found.")
        return
        
    if not os.path.exists("audit.csv"):
        print("audit.csv not found. Please run 'python audit.py' first.")
        return
        
    audit_data = {}
    with open("audit.csv", "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            audit_data[r["index"]] = r
            
    conf_table = {"approve": {"yes": 0, "no": 0}, "reject": {"yes": 0, "no": 0}, "manual_review": {"yes": 0, "no": 0}}
    false_approvals = []
    false_rejects = []
    valid_disposals = 0
    manual_reviews_for_valid = 0
    
    agreements = {"A": 0, "B_full1": 0, "B": 0, "C": 0}
    total_labels = 0
    weak_counts = {}
    
    with open(labels_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["index"]
            vd = row["valid_disposal"].strip().lower()
            if vd not in ["yes", "no"]:
                continue # blank or invalid
                
            total_labels += 1
            if idx not in audit_data:
                continue
                
            ar = audit_data[idx]
            v = ar["new_verdict"]
            
            if v in conf_table and vd in conf_table[v]:
                conf_table[v][vd] += 1
                
            if v == "approve" and vd == "no":
                false_approvals.append(idx)
            if v == "reject" and vd == "yes":
                false_rejects.append(idx)
                w = ar["weakest_check"]
                weak_counts[w] = weak_counts.get(w, 0) + 1
                
            if vd == "yes":
                valid_disposals += 1
                if v == "manual_review":
                    manual_reviews_for_valid += 1
                    
            f1_in = row["f1_item_in_hand"].strip().lower() == "yes"
            f2_bin = row["f2_bin_visible"].strip().lower() == "yes"
            f2_open = row["f2_open_palm_empty"].strip().lower() == "yes"
            
            A_pred = float(ar["new_A"]) >= 0.5
            B_pred = float(ar["new_B"]) >= 0.5
            Bf1_pred = float(ar.get("new_B_full1", 0)) >= 0.5
            C_pred = float(ar["new_C"]) >= 0.5
            
            if A_pred == f1_in: agreements["A"] += 1
            if B_pred == f2_bin: agreements["B"] += 1
            if C_pred == f2_open: agreements["C"] += 1
            
    print("=== LABEL MODE METRICS ===")
    print(f"Total labeled rows evaluated: {total_labels}")
    print("\nConfusion Table (Verdict \\ Valid Disposal):")
    print(f"                | YES | NO  |")
    print(f"----------------+-----+-----+")
    print(f" approve        | {conf_table['approve']['yes']:3d} | {conf_table['approve']['no']:3d} |")
    print(f" reject         | {conf_table['reject']['yes']:3d} | {conf_table['reject']['no']:3d} |")
    print(f" manual_review  | {conf_table['manual_review']['yes']:3d} | {conf_table['manual_review']['no']:3d} |")
    
    print(f"\nFalse Approvals (approve but vd=no): {false_approvals}")
    print(f"False Rejects (reject but vd=yes): {false_rejects}")
    print(f"Weakest checks among False Rejects: {weak_counts}")
    
    mr_rate = (manual_reviews_for_valid / valid_disposals * 100) if valid_disposals else 0
    print(f"Manual Review rate among Valid Disposals: {mr_rate:.1f}% ({manual_reviews_for_valid}/{valid_disposals})")
    
    print("\nPer-check agreement at threshold 0.5:")
    print(f" A vs f1_item_in_hand:      {(agreements['A']/max(1,total_labels))*100:.1f}%")
    print(f" B vs f2_bin_visible:       {(agreements['B']/max(1,total_labels))*100:.1f}%")
    print(f" C vs f2_open_palm_empty:   {(agreements['C']/max(1,total_labels))*100:.1f}%")
    
if __name__ == "__main__":
    main()
