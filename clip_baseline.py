import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from PIL import Image, UnidentifiedImageError

import torch
import open_clip

# Defined prompts for each class
CLASS_PROMPTS = {
    "cardboard": ["cardboard", "a cardboard box", "a piece of corrugated cardboard"],
    "glass": ["a glass bottle", "a glass jar", "broken glass"],
    "metal": ["a metal can", "an aluminium can", "a piece of scrap metal"],
    "paper": ["a piece of paper", "crumpled paper", "a newspaper"],
    "plastic": ["a plastic bottle", "plastic packaging", "a plastic wrapper"],
    "trash": ["general trash", "non-recyclable waste", "dirty garbage"]
}

EXPECTED_CLASSES = set(CLASS_PROMPTS.keys())
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def get_text_features(model, tokenizer, device):
    """
    Build text embeddings per class.
    Encodes each prompt as "a photo of {description}", L2-normalizes,
    averages across the class's prompts, and normalizes again.
    """
    class_features = []
    classes = list(CLASS_PROMPTS.keys())
    
    with torch.no_grad():
        for class_name in classes:
            prompts = [f"a photo of {desc}" for desc in CLASS_PROMPTS[class_name]]
            text_tokens = tokenizer(prompts).to(device)
            text_feats = model.encode_text(text_tokens)
            
            # L2 normalize each prompt's feature
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            
            # Average across prompts and normalize again
            text_feat_avg = text_feats.mean(dim=0, keepdim=True)
            text_feat_avg = text_feat_avg / text_feat_avg.norm(dim=-1, keepdim=True)
            class_features.append(text_feat_avg)
            
    # shape: (num_classes, feature_dim)
    text_features = torch.cat(class_features, dim=0)
    return classes, text_features

def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenCLIP zero-shot classification on TrashNet")
    parser.add_argument("--data", type=str, default="data/trashnet/dataset-resized", help="Path to dataset folder")
    parser.add_argument("--limit", type=int, default=0, help="Max images per class to evaluate (0 = all)")
    parser.add_argument("--batch", type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()

    data_dir = Path(args.data)
    
    if not data_dir.exists():
        print(f"Error: Dataset path '{data_dir}' does not exist.")
        return

    # Check for the expected class folders
    subdirs = [d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    found_classes = set(subdirs).intersection(EXPECTED_CLASSES)
    
    if len(found_classes) != len(EXPECTED_CLASSES):
        print(f"Error: Expected 6 class folders: {', '.join(EXPECTED_CLASSES)}.")
        print(f"Found: {', '.join(found_classes) if found_classes else 'none'}.")
        print("Check if you have a double-nested 'dataset-resized/dataset-resized' folder.")
        return

    # Setup device and model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print("Loading OpenCLIP model (ViT-B-32, laion2b_s34b_b79k)...")
    # ViT-B-32 with laion2b_s34b_b79k weights
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    print("Building text embeddings...")
    classes, text_features = get_text_features(model, tokenizer, device)

    # Collect image paths
    image_paths_by_class = {cls: [] for cls in classes}
    for cls in classes:
        cls_dir = data_dir / cls
        for file in cls_dir.iterdir():
            if file.is_file() and not file.name.startswith(".") and file.suffix.lower() in VALID_EXTENSIONS:
                image_paths_by_class[cls].append(file)
                
        # Sort for deterministic behavior
        image_paths_by_class[cls].sort()
        
        # Apply limit if specified
        if args.limit > 0:
            image_paths_by_class[cls] = image_paths_by_class[cls][:args.limit]

    # Flatten the dataset to (path, true_label)
    dataset = []
    for cls in classes:
        for path in image_paths_by_class[cls]:
            dataset.append((path, cls))

    print(f"Total images to process: {len(dataset)}")

    results = []
    mistakes = defaultdict(int)
    correct_count = 0
    correct_by_class = defaultdict(int)
    total_by_class = defaultdict(int)

    # Process in batches
    for i in range(0, len(dataset), args.batch):
        batch = dataset[i:i + args.batch]
        batch_images = []
        valid_indices = []
        
        for j, (path, true_label) in enumerate(batch):
            try:
                img = Image.open(path).convert("RGB")
                img_tensor = preprocess(img)
                batch_images.append(img_tensor)
                valid_indices.append(j)
            except (UnidentifiedImageError, OSError) as e:
                print(f"Warning: Skipping unreadable image '{path}' - {e}")
                
        if not batch_images:
            continue
            
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            image_feats = model.encode_image(batch_tensor)
            # L2 normalize image features
            image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True)
            
            # Compute similarity logits
            # softmax(100 * image_feats @ text_feats.T)
            logits = 100.0 * image_feats @ text_features.T
            probs = logits.softmax(dim=-1)
            
            confidences, predictions = probs.max(dim=-1)
            
        confidences = confidences.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        for idx, pred_idx, conf in zip(valid_indices, predictions, confidences):
            path, true_label = batch[idx]
            pred_label = classes[pred_idx]
            
            results.append((str(path), true_label, pred_label, float(conf)))
            
            total_by_class[true_label] += 1
            if true_label == pred_label:
                correct_count += 1
                correct_by_class[true_label] += 1
            else:
                mistakes[f"{true_label} -> {pred_label}"] += 1

    if not results:
        print("No valid images processed.")
        return

    # Print overall accuracy
    overall_acc = correct_count / len(results)
    print(f"\nOverall Accuracy: {overall_acc:.2%} ({correct_count}/{len(results)})")

    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    for cls in classes:
        total = total_by_class[cls]
        correct = correct_by_class[cls]
        acc = correct / total if total > 0 else 0
        print(f"  {cls}: {acc:.2%} ({correct}/{total})")

    # Print top mistakes
    print("\nTop 8 most common mistakes:")
    sorted_mistakes = sorted(mistakes.items(), key=lambda x: x[1], reverse=True)
    for mistake_str, count in sorted_mistakes[:8]:
        print(f"  {mistake_str}: {count}")

    # Save to CSV
    csv_path = "clip_baseline_results.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true", "pred", "confidence"])
        writer.writerows(results)
        
    print(f"\nPredictions saved to {csv_path}")

if __name__ == "__main__":
    main()
