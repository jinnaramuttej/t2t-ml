import argparse
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from t2t_validator import ClipScorer

def get_tiles(img, grid_size, crop_fraction=0.6):
    w, h = img.size
    crop_w, crop_h = int(w * crop_fraction), int(h * crop_fraction)
    
    if grid_size == 1:
        xs = [0]
        ys = [0]
    else:
        x_step = (w - crop_w) / (grid_size - 1)
        y_step = (h - crop_h) / (grid_size - 1)
        xs = [int(i * x_step) for i in range(grid_size)]
        ys = [int(i * y_step) for i in range(grid_size)]
        
    tiles = []
    for r, y in enumerate(ys):
        for c, x in enumerate(xs):
            tiles.append({
                "label": f"R{r+1}C{c+1}",
                "box": (x, y, x + crop_w, y + crop_h),
                "img": img.crop((x, y, x + crop_w, y + crop_h))
            })
    return tiles

def get_tap_crop(img, tap_x, tap_y, crop_fraction):
    w, h = img.size
    crop_w = w * crop_fraction
    crop_h = h * crop_fraction
    center_x = tap_x * w
    center_y = tap_y * h
    
    left = center_x - crop_w / 2
    top = center_y - crop_h / 2
    right = center_x + crop_w / 2
    bottom = center_y + crop_h / 2
    
    # Clamp
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
        
    left, top = max(0, left), max(0, top)
    right, bottom = min(w, right), min(h, bottom)
    
    return img.crop((left, top, right, bottom))

def main():
    parser = argparse.ArgumentParser(description="Diagnose CLIP scoring for specific images.")
    parser.add_argument("images", nargs="+", help="One or more image paths to diagnose")
    parser.add_argument("--tap", type=float, nargs=2, metavar=("X", "Y"), help="Optional tap coordinates (normalized 0 to 1)")
    args = parser.parse_args()

    print("Loading ClipScorer...")
    scorer = ClipScorer()
    device = scorer.device

    # Precompute individual phrase embeddings for has_object
    phrase_embs = {}
    with torch.no_grad():
        for group, phrases in scorer.phrases["has_object"].items():
            phrase_embs[group] = {}
            for p in phrases:
                text_tokens = scorer.tokenizer([f"a photo of {p}"]).to(device)
                feat = scorer.model.encode_text(text_tokens)
                feat = F.normalize(feat, dim=-1)
                phrase_embs[group][p] = feat

    for img_path in args.images:
        print(f"\n{'='*60}")
        print(f"Diagnosing: {img_path}")
        print(f"{'='*60}")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue

        # Compute full image features
        img_input = scorer.preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_feat = scorer.model.encode_image(img_input)
            img_feat = F.normalize(img_feat, dim=-1)

            # ---------------------------------------------------------
            # 1. Individual phrase scores (cosine sim * 100)
            # ---------------------------------------------------------
            print("\n--- 1. has_object Phrase Breakdown ---")
            best_phrase = None
            best_sim = -999.0
            
            print(f"{'Group':<10} | {'Score':<6} | Phrase")
            print("-" * 60)
            for group, p_dict in phrase_embs.items():
                for p, p_emb in p_dict.items():
                    sim = (100.0 * img_feat @ p_emb.T).item()
                    print(f"{group:<10} | {sim:5.1f}  | {p}")
                    if sim > best_sim:
                        best_sim = sim
                        best_phrase = p

            # Group probabilities
            holding_emb = scorer.text_embeddings["has_object"]["holding"]
            empty_emb = scorer.text_embeddings["has_object"]["empty"]
            no_hand_emb = scorer.text_embeddings["has_object"]["no_hand"]
            
            logits_3 = 100.0 * img_feat @ torch.cat([holding_emb, empty_emb, no_hand_emb], dim=0).T
            probs_3 = logits_3.softmax(dim=-1)[0].tolist()
            
            print(f"\nGroup Probabilities (3 groups):")
            print(f"  holding: {probs_3[0]:.4f}")
            print(f"  empty:   {probs_3[1]:.4f}")
            print(f"  no_hand: {probs_3[2]:.4f}")
            print(f"\nHighest single phrase: '{best_phrase}' ({best_sim:.1f})")

            # ---------------------------------------------------------
            # 2. Current score vs 2-group score
            # ---------------------------------------------------------
            print("\n--- 2. has_object: 3-group vs 2-group ---")
            print(f"Current score (3 groups): {probs_3[0]:.4f}")
            
            logits_2 = 100.0 * img_feat @ torch.cat([holding_emb, empty_emb], dim=0).T
            probs_2 = logits_2.softmax(dim=-1)[0].tolist()
            print(f"Score using only holding/empty: {probs_2[0]:.4f}")

            # ---------------------------------------------------------
            # 3. Grids for has_object "holding"
            # ---------------------------------------------------------
            print("\n--- 3. has_object Grid Analysis ---")
            for grid_size in [2, 3]:
                print(f"\n{grid_size}x{grid_size} Grid (60% crops):")
                tiles = get_tiles(img, grid_size, 0.6)
                scores = []
                
                print(f"  {'Tile':<6} | Score")
                print("  " + "-" * 20)
                for t in tiles:
                    t_score = scorer.has_object(t["img"])
                    scores.append((t_score, t["label"]))
                    print(f"  {t['label']:<6} | {t_score:.4f}")
                
                scores_only = [s[0] for s in scores]
                best_t = max(scores, key=lambda x: x[0])
                mean_s = sum(scores_only) / len(scores_only)
                print(f"\n  Mean: {mean_s:.4f} | Max: {best_t[0]:.4f} ({best_t[1]})")

            # ---------------------------------------------------------
            # 4. bin_present
            # ---------------------------------------------------------
            print("\n--- 4. bin_present Analysis ---")
            bp_full = scorer.bin_present(img)
            print(f"Full image score: {bp_full:.4f}")
            
            if args.tap:
                tx, ty = args.tap
                print(f"\nCrops at tap ({tx}, {ty}):")
                print(f"  {'Crop':<6} | Score")
                print("  " + "-" * 20)
                for frac in [0.4, 0.5, 0.7]:
                    c = get_tap_crop(img, tx, ty, frac)
                    sc = scorer.bin_present(c)
                    print(f"  {int(frac*100)}%    | {sc:.4f}")

if __name__ == "__main__":
    main()
