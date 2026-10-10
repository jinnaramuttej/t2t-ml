import json
import os
import unittest
from datetime import datetime

# Tests for backward compatibility and metadata parsing
class TestMetadataSchema(unittest.TestCase):
    
    def test_old_metadata_record(self):
        # Emulate an old metadata record that didn't have VLM or new fields
        old_meta = {
            "scores": {
                "A": 0.95, "B": 0.85, "C": 0.99, "B_full1": 0.90, "B_full2": 0.80, "item_still_in_hand": 0.01
            },
            "verdict": "approve",
            "test_type": "manual",
            "device": "iOS",
            "lat": 0.0,
            "lng": 0.0,
            "manual": False,
            "tap_point": {"x": 0.5, "y": 0.5},
            "f1_change": False,
            "vlm_passed": True,
            "duration_ms": None,
            "retries": 0
        }
        
        # Verify our export script handles it gracefully
        exported = convert_to_dataset_format("123", old_meta)
        self.assertEqual(exported["submission_id"], "123")
        self.assertIsNone(exported["vlm_verdict"])
        self.assertIsNone(exported["human_verified_label"])
        self.assertEqual(exported["pipeline_decision"], "approve")
        
    def test_missing_optional_fields(self):
        # Some fields might be missing entirely in very old records
        very_old = {
            "verdict": "reject"
        }
        exported = convert_to_dataset_format("456", very_old)
        self.assertEqual(exported["submission_id"], "456")
        self.assertEqual(exported["pipeline_decision"], "reject")
        self.assertIsNone(exported["sampling_strategy"])
        
    def test_malformed_vlm_output_handling(self):
        # Emulate how server.py would have saved a failed VLM response
        failed_meta = {
            "submission_id": "789",
            "verdict": "manual_review",
            "reason": "uncertain disposal (VLM)",
            "vlm_verdict": "ERROR",
            "vlm_note": "Failed to parse JSON"
        }
        exported = convert_to_dataset_format("789", failed_meta)
        self.assertEqual(exported["vlm_verdict"], "ERROR")
        self.assertEqual(exported["vlm_note"], "Failed to parse JSON")


def convert_to_dataset_format(submission_id, meta):
    """
    Converts a single submission's meta.json into a flat dictionary suitable for dataset generation.
    Handles legacy, missing, and new fields smoothly.
    """
    return {
        "submission_id": meta.get("submission_id", submission_id),
        "dataset_version": meta.get("dataset_version", "v1.0"),
        
        "model_id": meta.get("model_id"),
        "prompt_version": meta.get("prompt_version"),
        "sampling_strategy": meta.get("sampling_strategy"),
        "sampled_frame_timestamps": meta.get("sampled_frame_timestamps"),
        
        "vlm_verdict": meta.get("vlm_verdict"),
        "reason_code": meta.get("reason_code"),
        "vlm_note": meta.get("vlm_note"),
        
        "scores_A": meta.get("scores", {}).get("A"),
        "scores_B": meta.get("scores", {}).get("B"),
        "scores_C": meta.get("scores", {}).get("C"),
        
        "pipeline_decision": meta.get("verdict"),
        "reason": meta.get("reason"),
        "manual_review_status": meta.get("manual_review_status", meta.get("verdict") == "manual_review"),
        "human_verified_label": meta.get("human_verified_label"),
        
        "retry_count": meta.get("retries"),
        "duration_ms": meta.get("duration_ms")
    }


def export_dataset(submissions_dir="submissions", output_file="t2t_dataset_export.jsonl"):
    """
    Procedure for exporting reviewed examples into a training/evaluation dataset.
    This script reads all submissions, extracts the metadata, and saves it in a JSONL file.
    It deliberately avoids duplicating images/videos. To use the images, models can 
    reference the `submission_id` to look up the folder in the blob storage.
    """
    print(f"Exporting dataset from {submissions_dir} to {output_file}...")
    
    if not os.path.exists(submissions_dir):
        print(f"Directory {submissions_dir} not found. Skipping export.")
        return
        
    records = []
    
    for folder in os.listdir(submissions_dir):
        folder_path = os.path.join(submissions_dir, folder)
        if not os.path.isdir(folder_path):
            continue
            
        meta_path = os.path.join(folder_path, "meta.json")
        if not os.path.exists(meta_path):
            continue
            
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            # Only export records that have been human-verified!
            # if meta.get("human_verified_label") is None:
            #     continue 
            # (Commented out for demonstration purposes so it exports all)
            
            record = convert_to_dataset_format(folder, meta)
            records.append(record)
            
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            
    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Successfully exported {len(records)} records.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_dataset()
    else:
        unittest.main()
