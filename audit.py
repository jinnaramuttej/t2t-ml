import os
import sys
import json
import argparse
import statistics
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import imagehash
import csv

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
        
    left = max(0, left)
    top = max(0, top)
    right = min(w, right)
    bottom = min(h, bottom)
    
    return image.crop((left, top, right, bottom))

def classify_device(device_str):
    if not device_str:
        return "other"
    d = device_str.lower()
    if "; wv)" in d or "instagram" in d or "fban" in d or "fbav" in d or "whatsapp" in d or "line" in d:
        return "in-app or webview"
    if "chrome" in d:
        return "chrome"
    return "other"

def run_audit():
    base_dir = "submissions"
    if not os.path.exists(base_dir):
        print("No submissions folder found.")
        return
        
    folders = sorted(os.listdir(base_dir))
    subs = []
    skipped = []
    for f in folders:
        path = os.path.join(base_dir, f)
        if not os.path.isdir(path):
            continue
            
        f1_path = os.path.join(path, "frame1.jpg")
        f2_path = os.path.join(path, "frame2.jpg")
        meta_path = os.path.join(path, "meta.json")
        
        if not (os.path.exists(f1_path) and os.path.exists(f2_path) and os.path.exists(meta_path)):
            skipped.append((f, "Missing frame1, frame2, or meta.json"))
            continue
            
        subs.append(f)
        
    for s, reason in skipped:
        print(f"Skipped {s}: {reason}")
        
    print("Loading ClipScorer...")
    scorer = ClipScorer()
    
    results = []
    seen_hashes = []
    
    for idx, folder in enumerate(subs):
        index_str = f"{idx+1:02d}"
        path = os.path.join(base_dir, folder)
        f1_path = os.path.join(path, "frame1.jpg")
        f2_path = os.path.join(path, "frame2.jpg")
        meta_path = os.path.join(path, "meta.json")
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        img1 = Image.open(f1_path).convert("RGB")
        img2 = Image.open(f2_path).convert("RGB")
        
        tap_point = meta.get("tap_point", {})
        tap_x = tap_point.get("x")
        tap_y = tap_point.get("y")
        
        A = scorer.has_object(img1)
        if tap_x is not None and tap_y is not None:
            img1_crop = get_crop(img1, tap_x, tap_y)
            B = scorer.bin_present(img1_crop)
        else:
            B = scorer.bin_present(img1)
            
        C = scorer.hand_empty(img2)
        B_full1 = scorer.bin_present(img1)
        B_full2 = scorer.bin_present(img2)
        
        verdict, reason = decide(A, B, C, high=HIGH, low=LOW)
        weakest_check = reason.replace("weakest check: ", "") if "weakest check" in reason else ""
        
        lm1 = get_hand_landmarks(img1)
        hand_f1 = lm1 is not None
        
        lm2 = get_hand_landmarks(img2)
        hand_f2 = lm2 is not None
        hand_area_f2 = 0.0
        digits_f2 = 0
        if hand_f2:
            box2 = get_hand_box(landmarks_info=lm2)
            if box2:
                w2, h2 = img2.size
                hand_area_f2 = ((box2[2] - box2[0]) * (box2[3] - box2[1])) / (w2 * h2)
            digits_f2 = round(C * 5)
            
        h1 = imagehash.phash(img1)
        min_dist = 9999
        min_folder = None
        for past_h, past_folder in seen_hashes:
            d = h1 - past_h
            if d < min_dist:
                min_dist = d
                min_folder = past_folder
                
        if min_dist == 9999:
            min_dist = None
            
        seen_hashes.append((h1, folder))
        
        scores_dict = meta.get("scores", {})
        
        results.append({
            "index": index_str,
            "folder": folder,
            "timestamp": folder[:15],
            "test_type": meta.get("test_type", ""),
            "manual": meta.get("manual", False),
            "device_class": classify_device(meta.get("device", "")),
            "old_A": scores_dict.get("A", 0),
            "old_B": scores_dict.get("B", 0),
            "old_C": scores_dict.get("C", 0),
            "old_B_full1": scores_dict.get("B_full1", 0),
            "old_B_full2": scores_dict.get("B_full2", 0),
            "new_A": A,
            "new_B": B,
            "new_C": C,
            "new_B_full1": B_full1,
            "new_B_full2": B_full2,
            "new_verdict": verdict,
            "weakest_check": weakest_check,
            "tap_x": tap_x,
            "tap_y": tap_y,
            "hand_f1": hand_f1,
            "hand_f2": hand_f2,
            "hand_area_f2": hand_area_f2,
            "digits_f2": digits_f2,
            "min_hash_dist": min_dist,
            "min_hash_folder": min_folder,
            "img1": img1,
            "img2": img2
        })
        
    print(f"Processed {len(results)} submissions.")
    
    # 1. audit.csv
    with open("audit.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "folder", "time", "test_type", "manual", "device_class",
                         "old_A", "old_B", "old_C", "old_B_full1", "old_B_full2",
                         "new_A", "new_B", "new_C", "new_B_full1", "new_B_full2",
                         "new_verdict", "weakest_check", "tap_x", "tap_y",
                         "hand_f1", "hand_f2", "hand_area_f2", "digits_f2", "min_hash_dist"])
        for r in results:
            writer.writerow([r["index"], r["folder"], r["timestamp"], r["test_type"], r["manual"], r["device_class"],
                             r["old_A"], r["old_B"], r["old_C"], r["old_B_full1"], r["old_B_full2"],
                             r["new_A"], r["new_B"], r["new_C"], r["new_B_full1"], r["new_B_full2"],
                             r["new_verdict"], r["weakest_check"], r["tap_x"], r["tap_y"],
                             r["hand_f1"], r["hand_f2"], r["hand_area_f2"], r["digits_f2"], r["min_hash_dist"]])
                             
    # 2. audit_summary.txt
    verdicts = {}
    weakest = {}
    A, B, B_f1, B_f2, C_scores = [], [], [], [], []
    hands_f2 = 0
    manual_true = 0
    devices = {}
    
    for r in results:
        v = r["new_verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
        
        if v == "reject" and r["weakest_check"]:
            w = r["weakest_check"]
            weakest[w] = weakest.get(w, 0) + 1
            
        A.append(r["new_A"])
        B.append(r["new_B"])
        B_f1.append(r["new_B_full1"])
        B_f2.append(r["new_B_full2"])
        C_scores.append(r["new_C"])
        
        if r["hand_f2"]:
            hands_f2 += 1
            
        if r["manual"]:
            manual_true += 1
            
        d = r["device_class"]
        devices[d] = devices.get(d, 0) + 1
        
    lines = []
    lines.append("--- Verdicts ---")
    for k, v in verdicts.items(): lines.append(f"{k}: {v}")
    
    lines.append("\n--- Weakest Checks (Rejects) ---")
    for k, v in weakest.items(): lines.append(f"{k}: {v}")
    
    lines.append("\n--- Scores (Min, Median, Max) ---")
    for name, arr in [("A", A), ("B", B), ("B_full1", B_f1), ("B_full2", B_f2), ("C", C_scores)]:
        if arr:
            lines.append(f"{name}: {min(arr):.4f}, {statistics.median(arr):.4f}, {max(arr):.4f}")
            
    lines.append(f"\nHand detected in frame 2: {hands_f2}/{len(results)} ({hands_f2/max(1, len(results)):.1%})")
    lines.append(f"Manual=true count: {manual_true}")
    
    lines.append("\n--- Device Classes ---")
    for k, v in devices.items(): lines.append(f"{k}: {v}")
    
    lines.append("\n--- Hash Pairs (<=5) ---")
    for r in results:
        if r["min_hash_dist"] is not None and r["min_hash_dist"] <= 5:
            lines.append(f"{r['folder']} and {r['min_hash_folder']} (dist: {r['min_hash_dist']})")
            
    lines.append("\n--- Time Gaps (seconds) ---")
    gaps = []
    for i in range(1, len(results)):
        try:
            t1 = datetime.strptime(results[i-1]["timestamp"], "%Y%m%d_%H%M%S")
            t2 = datetime.strptime(results[i]["timestamp"], "%Y%m%d_%H%M%S")
            gaps.append(int((t2 - t1).total_seconds()))
        except ValueError:
            pass
    lines.append(f"{gaps}")
    
    content = "\n".join(lines)
    print("\n" + content)
    with open("audit_summary.txt", "w") as f:
        f.write(content)
        
    # 3 & 4. Sheets
    os.makedirs("audit_sheets", exist_ok=True)
    batch_size = 6
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
        
    for i in range(0, len(results), batch_size):
        batch = results[i:i+batch_size]
        
        def draw_sheet(is_scored):
            sheet_w = 800 if is_scored else 600
            sheet_h = 0
            row_heights = []
            
            for r in batch:
                w1, h1 = r["img1"].size
                h1_scaled = int(h1 * 240 / w1)
                row_h = h1_scaled + 20
                row_heights.append(row_h)
                sheet_h += row_h
                
            sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
            draw = ImageDraw.Draw(sheet)
            
            y_offset = 0
            for r, rh in zip(batch, row_heights):
                draw.text((10, y_offset + rh//2 - 10), str(r["index"]), fill="black", font=font)
                
                w1, h1 = r["img1"].size
                h1_scaled = int(h1 * 240 / w1)
                img1_resized = r["img1"].resize((240, h1_scaled))
                sheet.paste(img1_resized, (60, y_offset + 10))
                
                if r["tap_x"] is not None and r["tap_y"] is not None:
                    tx = 60 + int(r["tap_x"] * 240)
                    ty = y_offset + 10 + int(r["tap_y"] * h1_scaled)
                    r_rad = 5
                    draw.ellipse([tx-r_rad, ty-r_rad, tx+r_rad, ty+r_rad], outline="red", width=2)
                    
                w2, h2 = r["img2"].size
                h2_scaled = int(h2 * 240 / w2)
                img2_resized = r["img2"].resize((240, h2_scaled))
                sheet.paste(img2_resized, (310, y_offset + 10))
                
                if is_scored:
                    text_x = 560
                    info = [
                        f"A: {r['new_A']:.3f}",
                        f"B: {r['new_B']:.3f}",
                        f"B_f1: {r['new_B_full1']:.3f}",
                        f"B_f2: {r['new_B_full2']:.3f}",
                        f"C: {r['new_C']:.3f}",
                        f"Verdict: {r['new_verdict']}",
                        f"Weakest: {r['weakest_check']}"
                    ]
                    draw.text((text_x, y_offset + 10), "\n".join(info), fill="black", font=font)
                    
                y_offset += rh
                
            return sheet
            
        page_num = (i // batch_size) + 1
        blind_sheet = draw_sheet(False)
        blind_sheet.save(f"audit_sheets/blind_{page_num:02d}.jpg", quality=80)
        
        scored_sheet = draw_sheet(True)
        scored_sheet.save(f"audit_sheets/scored_{page_num:02d}.jpg", quality=80)
        
    # 5. Label template
    with open("audit_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "folder", "f1_item_in_hand", "f2_bin_visible", "f2_open_palm_empty", "valid_disposal", "fake_type", "notes"])
        for r in results:
            writer.writerow([r["index"], r["folder"], "", "", "", "", "", ""])

def run_labels_mode(labels_file):
    if not os.path.exists("audit.csv"):
        print("audit.csv not found. Please run without --labels first.")
        return
        
    audit_data = {}
    with open("audit.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audit_data[row["index"]] = row
            
    conf_table = {"yes": {"approve": 0, "reject": 0, "manual_review": 0}, 
                  "no": {"approve": 0, "reject": 0, "manual_review": 0}}
    false_approvals = []
    false_rejects = []
    
    A_match, B_f1_match, B_match, C_match = 0, 0, 0, 0
    A_total, B_f1_total, B_total, C_total = 0, 0, 0, 0
    
    valid_count = 0
    valid_manual = 0
    
    weakest_false_rejects = {}
    
    with open(labels_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["index"]
            valid = row.get("valid_disposal", "").strip().lower()
            if not valid:
                continue
                
            if idx not in audit_data:
                continue
                
            aud = audit_data[idx]
            verdict = aud["new_verdict"]
            
            if valid in conf_table:
                conf_table[valid][verdict] += 1
                
            if valid == "yes":
                valid_count += 1
                if verdict == "reject":
                    false_rejects.append(idx)
                    w = aud["weakest_check"]
                    if w:
                        weakest_false_rejects[w] = weakest_false_rejects.get(w, 0) + 1
                elif verdict == "manual_review":
                    valid_manual += 1
            elif valid == "no":
                if verdict == "approve":
                    false_approvals.append(idx)
                    
            f1_item = row.get("f1_item_in_hand", "").strip().lower()
            f2_bin = row.get("f2_bin_visible", "").strip().lower()
            f2_palm = row.get("f2_open_palm_empty", "").strip().lower()
            
            a_val = float(aud["new_A"])
            b_f1_val = float(aud["new_B_full1"])
            b_val = float(aud["new_B"])
            c_val = float(aud["new_C"])
            
            if f1_item in ["yes", "no"]:
                A_total += 1
                if (a_val >= 0.5 and f1_item == "yes") or (a_val < 0.5 and f1_item == "no"):
                    A_match += 1
                    
            if f2_bin in ["yes", "no"]:
                B_f1_total += 1
                if (b_f1_val >= 0.5 and f2_bin == "yes") or (b_f1_val < 0.5 and f2_bin == "no"):
                    B_f1_match += 1
                    
                B_total += 1
                if (b_val >= 0.5 and f2_bin == "yes") or (b_val < 0.5 and f2_bin == "no"):
                    B_match += 1
                    
            if f2_palm in ["yes", "no"]:
                C_total += 1
                if (c_val >= 0.5 and f2_palm == "yes") or (c_val < 0.5 and f2_palm == "no"):
                    C_match += 1
                    
    print("\n--- Confusion Table (valid_disposal vs verdict) ---")
    print(f"valid=yes -> approve: {conf_table['yes']['approve']}, reject: {conf_table['yes']['reject']}, manual_review: {conf_table['yes']['manual_review']}")
    print(f"valid=no  -> approve: {conf_table['no']['approve']}, reject: {conf_table['no']['reject']}, manual_review: {conf_table['no']['manual_review']}")
    
    print(f"\nFalse Approvals (valid=no, verdict=approve): {false_approvals}")
    print(f"False Rejects (valid=yes, verdict=reject): {false_rejects}")
    
    if valid_count > 0:
        print(f"\nManual Review Rate (among valid disposals): {valid_manual / valid_count:.2%} ({valid_manual}/{valid_count})")
        
    print("\n--- Per-Check Agreement at 0.5 ---")
    if A_total: print(f"A vs f1_item_in_hand: {A_match/A_total:.2%} ({A_match}/{A_total})")
    if B_f1_total: print(f"B_full1 vs f2_bin_visible: {B_f1_match/B_f1_total:.2%} ({B_f1_match}/{B_f1_total})")
    if B_total: print(f"B vs f2_bin_visible: {B_match/B_total:.2%} ({B_match}/{B_total})")
    if C_total: print(f"C vs f2_open_palm_empty: {C_match/C_total:.2%} ({C_match}/{C_total})")
    
    print("\n--- Weakest Check among False Rejects ---")
    for k, v in weakest_false_rejects.items():
        print(f"{k}: {v}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, help="Run in labels mode with the given labels CSV")
    args = parser.parse_args()
    
    if args.labels:
        run_labels_mode(args.labels)
    else:
        run_audit()

if __name__ == "__main__":
    main()
