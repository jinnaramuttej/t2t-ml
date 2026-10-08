import argparse
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from t2t_validator import ClipScorer

def analyze_image(img_path, scorer):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"Failed to load {img_path}: {e}")
        return

    print(f"\n{'='*60}")
    print(f"Diagnosing: {img_path}")
    print(f"{'='*60}")

    # 1. Current has_object (A)
    current_A = scorer.has_object(img)

    # 2. Re-scored has_object (A) with "a dustbin with no hand in view" removed
    custom_phrases = {
        "holding": ["a hand holding an object", "a hand holding a piece of trash"],
        "empty": ["an empty hand", "an open palm with nothing in it"],
        "no_hand": ["a floor", "a wall"]
    }
    
    with scorer.torch.no_grad(), scorer.torch.amp.autocast('cuda' if scorer.device == 'cuda' else 'cpu'):
        holding_emb = scorer._embed_and_average_phrases(custom_phrases["holding"])
        empty_emb = scorer._embed_and_average_phrases(custom_phrases["empty"])
        no_hand_emb = scorer._embed_and_average_phrases(custom_phrases["no_hand"])

        text_features = scorer.torch.cat([holding_emb, empty_emb, no_hand_emb], dim=0)
        
        image_input = scorer.preprocess(img).unsqueeze(0).to(scorer.device)
        image_features = scorer.model.encode_image(image_input)
        image_features = F.normalize(image_features, dim=-1)

        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1)
        rescored_A = probs[0, 0].item()

    # 3. Current bin_present (B)
    current_B = scorer.bin_present(img)

    print(f"1. Current has_object (A):     {current_A:.4f}")
    print(f"2. Re-scored has_object (A):   {rescored_A:.4f} (removed 'dustbin' phrase from no_hand)")
    print(f"3. Current bin_present (B):    {current_B:.4f}\n")

    # 4. bin_present (B) on grids
    def evaluate_grid(grid_size):
        w, h = img.size
        tile_w = int(w * 0.6)
        tile_h = int(h * 0.6)
        
        x_starts = [int(i * (w - tile_w) / (grid_size - 1)) for i in range(grid_size)]
        y_starts = [int(i * (h - tile_h) / (grid_size - 1)) for i in range(grid_size)]
        
        results = []
        for r, y in enumerate(y_starts):
            for c, x in enumerate(x_starts):
                crop = img.crop((x, y, x + tile_w, y + tile_h))
                score = scorer.bin_present(crop)
                results.append(((r, c), score))
                
        return results

    for grid_size in [2, 3]:
        print(f"4. bin_present (B) on {grid_size}x{grid_size} grid (60% overlapping crops):")
        results = evaluate_grid(grid_size)
        
        print(f"   {'-'*35}")
        print(f"   | {'Tile (r, c)':<15} | {'Score':<10} |")
        print(f"   {'-'*35}")
        
        scores = []
        max_score = -1.0
        max_tile = None
        for tile, score in results:
            scores.append(score)
            if score > max_score:
                max_score = score
                max_tile = tile
            print(f"   | {str(tile):<15} | {score:<10.4f} |")
        
        print(f"   {'-'*35}")
        mean_score = sum(scores) / len(scores)
        print(f"   Max:  {max_score:.4f} at Tile {max_tile}")
        print(f"   Mean: {mean_score:.4f}\n")

def main():
    parser = argparse.ArgumentParser(description="Diagnose image with custom CLIP parameters.")
    parser.add_argument("images", nargs='+', help="Paths to images")
    args = parser.parse_args()
    
    print("Loading ClipScorer...")
    scorer = ClipScorer()
    
    for img_path in args.images:
        analyze_image(img_path, scorer)

if __name__ == "__main__":
    main()
