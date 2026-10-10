import os
import json
import csv
from PIL import Image
import imagehash
import torch
import torch.nn.functional as F
import sys

# We will import modules from t2t_validator.py for baseline
from t2t_validator import ClipScorer, get_hand_landmarks, get_hand_box, get_padded_hand_crop

def decide_new(a, b, c, is_dup=False, gps_ok=True, high=0.75, low=0.35):
    """
    Proposed decide logic where `hand_empty` (c) is a supporting signal.
    """
    if is_dup:
        return "reject", "duplicate frames"
    if not gps_ok:
        return "reject", "implausible or mocked location"
        
    score = min(a, b) # c is excluded from hard reject
    if score <= low:
        weakest_val = a
        weakest_name = "has_object"
        if b < weakest_val:
            weakest_val = b
            weakest_name = "bin_present"
        return "reject", f"weakest check: {weakest_name}"
        
    # a and b are > low
    if score >= high:
        # Before approving, check the supporting signal
        if c <= low:
            return "manual_review", "hand_empty check failed (possible motion blur or fist)"
        return "approve", ""
    else:
        return "manual_review", ""

class ImprovedClipScorer(ClipScorer):
    def __init__(self):
        super().__init__()
        # Redefine phrases for empty fist detection
        self.phrases = {
            "has_object": {
                "holding": ["a hand holding an object", "a hand holding a piece of trash"],
                "empty": ["an empty hand", "an open palm with nothing in it"],
                "no_hand": ["a floor", "a wall"]
            },
            "is_empty_fist": {
                "fist": ["a closed fist", "an empty fist with no object", "a hand curled into a fist"],
                "other": ["a hand holding an object", "a flat open palm", "a piece of trash"]
            },
            "bin_present": {
                "pos": ["a dustbin", "a trash can", "a garbage bin", "a waste container"],
                "neg": ["a floor", "a wall", "a road", "a table"]
            }
        }
        # Re-cache text embeddings
        with self.torch.no_grad():
            for check_name, groups in self.phrases.items():
                self.text_embeddings[check_name] = {}
                for group_name, phrases in groups.items():
                    self.text_embeddings[check_name][group_name] = self._embed_and_average_phrases(phrases)
                    
    def is_empty_fist(self, image):
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            fist_emb = self.text_embeddings["is_empty_fist"]["fist"]
            other_emb = self.text_embeddings["is_empty_fist"]["other"]
            text_features = self.torch.cat([fist_emb, other_emb], dim=0)
            logits = 100.0 * image_features @ text_features.T
            probs = logits.softmax(dim=-1)
            return probs[0, 0].item()
            
    def get_image_embedding(self, image):
        """Returns the L2-normalized image embedding for advanced deduplication."""
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(image_input)
            return F.normalize(image_features, dim=-1)

def main():
    print("Loading submissions...")
    # Load labels to know truth
    labels = {}
    with open("audit_labels.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['folder']] = row
            
    folders = sorted([f for f in os.listdir("submissions") if os.path.isdir(os.path.join("submissions", f))])
    
    print("Initializing models...")
    baseline_scorer = ClipScorer()
    improved_scorer = ImprovedClipScorer()
    
    submissions = []
    
    for folder in folders:
        folder_path = os.path.join("submissions", folder)
        meta_path = os.path.join(folder_path, "meta.json")
        f1_path = os.path.join(folder_path, "frame1.jpg")
        f2_path = os.path.join(folder_path, "frame2.jpg")
        
        if not os.path.exists(meta_path) or not os.path.exists(f1_path) or not os.path.exists(f2_path):
            continue
            
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        img1 = Image.open(f1_path).convert('RGB')
        img2 = Image.open(f2_path).convert('RGB')
        
        # We need the crop for img1
        tap_point = meta.get("tap_point", {"x": 0.5, "y": 0.5})
        w, h = img1.size
        cx, cy = int(tap_point["x"] * w), int(tap_point["y"] * h)
        crop_size = 224
        x1, y1 = max(0, cx - crop_size//2), max(0, cy - crop_size//2)
        x2, y2 = min(w, cx + crop_size//2), min(h, cy + crop_size//2)
        crop1 = img1.crop((x1, y1, x2, y2))
        
        submissions.append({
            "folder": folder,
            "meta": meta,
            "img1": img1,
            "crop1": crop1,
            "img2": img2,
            "label": labels.get(folder, {})
        })
        
    print(f"Loaded {len(submissions)} submissions.")
    
    print("\n--- 1. Evaluating hand_empty logic change ---")
    # Submission 20 was falsely rejected due to C = 0.2
    # Let's see if the new logic fixes it.
    sub20 = next(s for s in submissions if s["folder"].startswith("20261008_182957"))
    A, B, C = sub20["meta"]["scores"]["A"], sub20["meta"]["scores"]["B"], sub20["meta"]["scores"]["C"]
    new_verdict, new_reason = decide_new(A, B, C)
    print(f"Submission 20 (Motion Blur Genuine): Old Verdict = reject. New Verdict = {new_verdict} (Reason: {new_reason})")
    
    print("\n--- 2. Evaluating Empty Fist Detection (Submission 19) ---")
    sub19 = next(s for s in submissions if s["folder"].startswith("20261008_182939"))
    img1_19 = sub19["img1"]
    
    old_A = baseline_scorer.has_object(img1_19)
    new_A = improved_scorer.has_object(img1_19)
    fist_score = improved_scorer.is_empty_fist(img1_19)
    print(f"Submission 19 (Empty Fist Attack): Baseline has_object = {old_A:.4f}")
    print(f"  -> New Dedicated is_empty_fist score = {fist_score:.4f}")
    
    # Check normal holding object to ensure it doesn't trigger false positives
    sub08 = next(s for s in submissions if s["folder"].startswith("20261008_140713"))
    fist_score_genuine = improved_scorer.is_empty_fist(sub08["img1"])
    print(f"Submission 08 (Genuine Disposal): is_empty_fist score = {fist_score_genuine:.4f}")
    
    print("\n--- 3. Evaluating Deduplication Trade-offs ---")
    # Compute all hashes and embeddings
    for s in submissions:
        s["h1"] = imagehash.phash(s["img1"])
        s["h2"] = imagehash.phash(s["img2"])
        s["emb1"] = improved_scorer.get_image_embedding(s["img1"])
        s["emb2"] = improved_scorer.get_image_embedding(s["img2"])
        
    # Compute pairwise distances
    n = len(submissions)
    hash_distances = []
    emb_distances = []
    
    for i in range(n):
        for j in range(i+1, n):
            h1_dist = submissions[i]["h1"] - submissions[j]["h1"]
            # Embedding distance: 1 - cosine_similarity
            cos_sim = F.cosine_similarity(submissions[i]["emb1"], submissions[j]["emb1"]).item()
            emb_dist = 1.0 - cos_sim
            
            # Print if they are very close
            if h1_dist < 20 or emb_dist < 0.1:
                print(f"Pairs ({submissions[i]['folder'][:15]}, {submissions[j]['folder'][:15]}): Hash Dist = {h1_dist}, Emb Dist = {emb_dist:.4f}")
                
    # Sub 17 and 18 are "wrong_bin" attacks, maybe same bin? Let's check their distance
    sub17 = next(s for s in submissions if s["folder"].startswith("20261008_182844"))
    sub18 = next(s for s in submissions if s["folder"].startswith("20261008_182925"))
    h_dist = sub17["h1"] - sub18["h1"]
    e_dist = 1.0 - F.cosine_similarity(sub17["emb1"], sub18["emb1"]).item()
    print(f"\nSubmissions 17 & 18 (Reused Scene Check): Hash Dist = {h_dist}, Emb Dist = {e_dist:.4f}")
    
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
