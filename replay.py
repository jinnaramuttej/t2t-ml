import argparse
import json
import os
import sys
from PIL import Image

# Import the refactored score_frames from server
import server

def main():
    parser = argparse.ArgumentParser(description="Replay scoring on a past submission.")
    parser.add_argument("folder", help="Path to the submission folder (e.g., submissions/2026...)")
    args = parser.parse_args()

    folder = args.folder
    meta_path = os.path.join(folder, "meta.json")
    f1_path = os.path.join(folder, "frame1.jpg")
    f2_path = os.path.join(folder, "frame2.jpg")

    if not os.path.exists(meta_path):
        print(f"Error: {meta_path} not found")
        sys.exit(1)
        
    if not os.path.exists(f1_path) or not os.path.exists(f2_path):
        print("Error: frame1.jpg or frame2.jpg not found in the folder")
        sys.exit(1)

    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    tap = meta.get("tap_point", {})
    tap_x = tap.get("x")
    tap_y = tap.get("y")
    
    img1 = Image.open(f1_path).convert("RGB")
    img2 = Image.open(f2_path).convert("RGB")
    
    print("Loading ClipScorer...")
    # Initialize the global scorer in the server module since we aren't starting the FastAPI server
    server.scorer = server.ClipScorer()
    
    print("Scoring frames...")
    A, B, C, B_full1, B_full2, item_still_in_hand = server.score_frames(img1, img2, tap_x, tap_y)
    
    old = meta.get("scores", {})
    
    keys = ["A", "B", "C", "B_full1", "B_full2", "item_still_in_hand"]
    new_scores = [A, B, C, B_full1, B_full2, item_still_in_hand]
    
    print(f"\n{'Metric':<20} | {'Old':<8} | {'New':<8} | {'Diff':<8}")
    print("-" * 53)
    for k, new_v in zip(keys, new_scores):
        old_v = old.get(k, 0.0)
        diff = new_v - old_v
        print(f"{k:<20} | {old_v:8.4f} | {new_v:8.4f} | {diff:+8.4f}")

if __name__ == "__main__":
    main()
