"""
t2t_validator.py
Validates trash-disposal video clips using OpenCLIP in a zero-shot manner.
Takes two frames (frame 1 = user holding an item, frame 2 = item gone and a dustbin visible)
and returns a verdict: approve, reject, or manual_review.

Install dependencies:
pip install mediapipe torch torchvision open_clip_torch pillow imagehash opencv-python
"""

import argparse
import sys

# These thresholds must be tuned on labeled clips for optimal performance.
HIGH = 0.75
LOW = 0.35

def decide(a, b, c, is_dup=False, gps_ok=True, high=HIGH, low=LOW):
    """
    Decision logic for the validator.
    Returns (verdict, reason).
    """
    if is_dup:
        return "reject", "duplicate frames"
    if not gps_ok:
        return "reject", "implausible or mocked location"
    
    score = min(a, b, c)
    if score >= high:
        return "approve", ""
    elif score <= low:
        # Identify the weakest check
        weakest_val = a
        weakest_name = "has_object"
        if b < weakest_val:
            weakest_val = b
            weakest_name = "bin_present"
        if c < weakest_val:
            weakest_val = c
            weakest_name = "hand_empty"
        return "reject", f"weakest check: {weakest_name}"
    else:
        return "manual_review", ""

def is_duplicate(image, seen_hashes, max_distance=5):
    """
    Checks if an image is a duplicate based on perceptual hashing.
    image: PIL.Image
    seen_hashes: list of imagehash.ImageHash objects
    max_distance: maximum hamming distance to be considered a duplicate
    """
    import imagehash # lazy import
    current_hash = imagehash.phash(image)
    for h in seen_hashes:
        if current_hash - h <= max_distance:
            return True
    return False

HAND_PAD = 0.4
_HANDS_DETECTOR = None

def get_hand_landmarks(image):
    """
    Takes a PIL image and returns (landmarks, w, h) or None if no hand is found.
    Uses the mediapipe Tasks HandLandmarker API.
    """
    global _HANDS_DETECTOR
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    import numpy as np
    import os
    import sys
    
    if _HANDS_DETECTOR is None:
        model_path = "models/hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Error: HandLandmarker model not found.")
            print(f"Please download it from the MediaPipe Hand Landmarker Python guide and save it to '{model_path}'.")
            sys.exit(1)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5
        )
        _HANDS_DETECTOR = vision.HandLandmarker.create_from_options(options)
        
    image_np = np.array(image.convert("RGB"))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
    
    results = _HANDS_DETECTOR.detect(mp_image)
    
    if not results.hand_landmarks:
        return None
        
    landmarks = results.hand_landmarks[0]
    h, w, _ = image_np.shape
    return (landmarks, w, h)

def get_hand_box(image=None, landmarks_info=None):
    """
    Takes a PIL image or landmarks_info and returns the hand's bounding box 
    (from landmarks) as (xmin, ymin, xmax, ymax), or None if no hand is found.
    """
    if landmarks_info is None:
        if image is None:
            return None
        landmarks_info = get_hand_landmarks(image)
        
    if landmarks_info is None:
        return None
        
    landmarks, w, h = landmarks_info
    
    x_min = w
    y_min = h
    x_max = 0
    y_max = 0
    
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)
        
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(w, x_max)
    y_max = min(h, y_max)
        
    return (x_min, y_min, x_max, y_max)

def get_padded_hand_crop(image, box):
    if box is None:
        return None
    x_min, y_min, x_max, y_max = box
    w, h = image.size
    
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    pad_w = int(box_w * HAND_PAD)
    pad_h = int(box_h * HAND_PAD)
    
    new_x_min = max(0, x_min - pad_w)
    new_y_min = max(0, y_min - pad_h)
    new_x_max = min(w, x_max + pad_w)
    new_y_max = min(h, y_max + pad_h)
    
    return image.crop((new_x_min, new_y_min, new_x_max, new_y_max))

class ClipScorer:
    def __init__(self):
        import torch
        import open_clip
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load OpenCLIP ViT-B-32, pretrained laion2b_s34b_b79k
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=self.device, precision='fp32')
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.torch = torch
        
        # Define phrases
        self.phrases = {
            "has_object": {
                "holding": ["a hand holding an object", "a hand holding a piece of trash"],
                "empty": ["an empty hand", "an open palm with nothing in it"],
                "no_hand": ["a floor", "a wall"] # removed "a dustbin with no hand in view"
            },
            "bin_present": {
                "pos": ["a dustbin", "a trash can", "a garbage bin", "a waste container"],
                "neg": ["a floor", "a wall", "a road", "a table"]
            }
        }
        
        # Cache text embeddings
        self.text_embeddings = {}
        with self.torch.no_grad():
            for check_name, groups in self.phrases.items():
                self.text_embeddings[check_name] = {}
                for group_name, phrases in groups.items():
                    self.text_embeddings[check_name][group_name] = self._embed_and_average_phrases(phrases)
                    
    def _embed_and_average_phrases(self, phrases):
        """
        Wraps phrases as 'a photo of {phrase}', embeds them, L2-normalizes, 
        and averages them into one normalized vector.
        """
        import torch.nn.functional as F
        
        texts = [f"a photo of {phrase}" for phrase in phrases]
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
        
        # Average into one vector and re-normalize
        avg_feature = text_features.mean(dim=0, keepdim=True)
        avg_feature = F.normalize(avg_feature, dim=-1)
        return avg_feature
        
    def _score(self, image, check_name):
        """
        Scores an image against a specific check.
        Returns probability of the positive/target group.
        """
        import torch.nn.functional as F
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = F.normalize(image_features, dim=-1)
            
            if check_name == "has_object":
                holding_emb = self.text_embeddings[check_name]["holding"]
                empty_emb = self.text_embeddings[check_name]["empty"]
                no_hand_emb = self.text_embeddings[check_name]["no_hand"]
                
                text_features = self.torch.cat([holding_emb, empty_emb, no_hand_emb], dim=0)
                logits = 100.0 * image_features @ text_features.T
                probs = logits.softmax(dim=-1)
                
                return probs[0, 0].item()
            else:
                pos_emb = self.text_embeddings[check_name]["pos"] # shape (1, D)
                neg_emb = self.text_embeddings[check_name]["neg"] # shape (1, D)
                
                # shape (2, D)
                text_features = self.torch.cat([pos_emb, neg_emb], dim=0)
                
                # Compute softmax(100 * image_feat @ [pos, neg].T)
                logits = 100.0 * image_features @ text_features.T
                probs = logits.softmax(dim=-1)
                
                # Return probability of the positive group (index 0)
                return probs[0, 0].item()

    def has_object(self, image):
        return self._score(image, "has_object")

    def bin_present(self, image):
        return self._score(image, "bin_present")

    def hand_empty(self, image):
        self.last_hand_crop = None
        landmarks_info = get_hand_landmarks(image)
        if landmarks_info is None:
            return 0.5
            
        landmarks, w, h = landmarks_info
        box = get_hand_box(landmarks_info=landmarks_info)
        self.last_hand_crop = get_padded_hand_crop(image, box)
        
        import math
        # ensure finger-extension distances use pixel coordinates (x*w, y*h) not normalized values
        def dist(lm1, lm2):
            return math.hypot((lm1.x * w) - (lm2.x * w), (lm1.y * h) - (lm2.y * h))
            
        extended_count = 0
        
        # Index: 0 to 8 > 0 to 6
        if dist(landmarks[0], landmarks[8]) > dist(landmarks[0], landmarks[6]):
            extended_count += 1
        # Middle: 0 to 12 > 0 to 10
        if dist(landmarks[0], landmarks[12]) > dist(landmarks[0], landmarks[10]):
            extended_count += 1
        # Ring: 0 to 16 > 0 to 14
        if dist(landmarks[0], landmarks[16]) > dist(landmarks[0], landmarks[14]):
            extended_count += 1
        # Pinky: 0 to 20 > 0 to 18
        if dist(landmarks[0], landmarks[20]) > dist(landmarks[0], landmarks[18]):
            extended_count += 1
        # Thumb: 17 to 4 > 17 to 3
        if dist(landmarks[17], landmarks[4]) > dist(landmarks[17], landmarks[3]):
            extended_count += 1
            
        return extended_count / 5.0

def run_selftest():
    """
    Runs assertions on decide() without loading models.
    """
    print("Running selftest on decide()...")
    
    # Approve case
    verdict, reason = decide(0.8, 0.9, 0.85)
    assert verdict == "approve", f"Expected approve, got {verdict}"
    assert reason == "", f"Expected empty reason, got {reason}"
    
    # Reject case (min score <= LOW)
    verdict, reason = decide(0.8, 0.3, 0.9)
    assert verdict == "reject", f"Expected reject, got {verdict}"
    assert "bin_present" in reason, f"Expected bin_present in reason, got {reason}"
    
    # Manual review case (LOW < min score < HIGH)
    verdict, reason = decide(0.8, 0.6, 0.9)
    assert verdict == "manual_review", f"Expected manual_review, got {verdict}"
    
    # Duplicate case
    verdict, reason = decide(0.9, 0.9, 0.9, is_dup=True)
    assert verdict == "reject", f"Expected reject, got {verdict}"
    assert "duplicate" in reason, f"Expected duplicate in reason, got {reason}"
    
    # Bad-GPS case
    verdict, reason = decide(0.9, 0.9, 0.9, gps_ok=False)
    assert verdict == "reject", f"Expected reject, got {verdict}"
    assert "location" in reason, f"Expected location in reason, got {reason}"
    
    print("Selftest passed successfully.")

def main():
    parser = argparse.ArgumentParser(description="T2T Validator: zero-shot trash-disposal video validation.")
    parser.add_argument("--frame1", type=str, help="Path to frame 1 (user holding item)")
    parser.add_argument("--frame2", type=str, help="Path to frame 2 (item gone, dustbin visible)")
    parser.add_argument("--hand-test", type=str, help="Path to an image to test hand detection")
    parser.add_argument("--selftest", action="store_true", help="Run selftest on decision logic without loading model")
    
    args = parser.parse_args()
    
    if args.hand_test:
        from PIL import Image
        try:
            img = Image.open(args.hand_test).convert("RGB")
        except Exception as e:
            print(f"Error loading image: {e}")
            sys.exit(1)
            
        landmarks_info = get_hand_landmarks(img)
        if landmarks_info is None:
            print("no hand detected")
        else:
            landmarks, w, h = landmarks_info
            box = get_hand_box(landmarks_info=landmarks_info)
            print(f"Hand detected at: {box}")
            
            import math
            # ensure finger-extension distances use pixel coordinates (x*w, y*h) not normalized values
            def dist(lm1, lm2):
                return math.hypot((lm1.x * w) - (lm2.x * w), (lm1.y * h) - (lm2.y * h))
                
            extended_digits = []
            if dist(landmarks[17], landmarks[4]) > dist(landmarks[17], landmarks[3]):
                extended_digits.append("thumb")
            if dist(landmarks[0], landmarks[8]) > dist(landmarks[0], landmarks[6]):
                extended_digits.append("index")
            if dist(landmarks[0], landmarks[12]) > dist(landmarks[0], landmarks[10]):
                extended_digits.append("middle")
            if dist(landmarks[0], landmarks[16]) > dist(landmarks[0], landmarks[14]):
                extended_digits.append("ring")
            if dist(landmarks[0], landmarks[20]) > dist(landmarks[0], landmarks[18]):
                extended_digits.append("pinky")
                
            fraction = len(extended_digits) / 5.0
            print(f"Extended digits: {', '.join(extended_digits) if extended_digits else 'none'}")
            print(f"Extended fraction: {fraction:.2f}")
            
            crop = get_padded_hand_crop(img, box)
            if crop is not None:
                crop.save("hand_test_crop.jpg")
                print("Crop saved to hand_test_crop.jpg")
        sys.exit(0)
        
    if args.selftest:
        run_selftest()
        sys.exit(0)
        
    if args.frame1 and args.frame2:
        # Lazy imports for actual inference
        from PIL import Image
        
        try:
            img1 = Image.open(args.frame1).convert('RGB')
            img2 = Image.open(args.frame2).convert('RGB')
        except Exception as e:
            print(f"Error loading images: {e}")
            sys.exit(1)
            
        print("Loading OpenCLIP model...")
        scorer = ClipScorer()
        
        print(f"Scoring {args.frame1} for 'has_object'...")
        score_has_object = scorer.has_object(img1)
        print(f"Scoring {args.frame2} for 'bin_present'...")
        score_bin_present = scorer.bin_present(img2)
        print(f"Scoring {args.frame2} for 'hand_empty'...")
        score_hand_empty = scorer.hand_empty(img2)
        
        print("\n--- Scores ---")
        print(f"has_object (frame1):  {score_has_object:.4f}")
        print(f"bin_present (frame2): {score_bin_present:.4f}")
        print(f"hand_empty (frame2):  {score_hand_empty:.4f}")
        
        verdict, reason = decide(score_has_object, score_bin_present, score_hand_empty)
        
        print("\n--- Verdict ---")
        print(f"Verdict: {verdict}")
        if reason:
            print(f"Reason:  {reason}")
            
    elif not args.selftest:
        parser.print_help()

if __name__ == "__main__":
    main()
