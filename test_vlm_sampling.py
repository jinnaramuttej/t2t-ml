import cv2
import numpy as np
import tempfile
import os
import unittest

def sample_video_frames(video_bytes):
    if not video_bytes:
        return [], "missing_video"
        
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name
        
    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    timestamps = []
    
    while True:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        ret, frame = cap.read()
        if not ret:
            break
        # OpenCV sometimes reports invalid (0.0) timestamps for webm
        if not timestamps or pos_msec > timestamps[-1]:
            timestamps.append(pos_msec)
        else:
            # Fallback for invalid/stuck timestamps: assume 33ms per frame (~30fps)
            timestamps.append(timestamps[-1] + 33.3)
            
        frames.append(cv2.resize(frame, (128, 128))) # Keep small for testing memory
        
    cap.release()
    os.remove(temp_video_path)
    
    n = len(frames)
    if n == 0:
        return [], "empty_video"
        
    strategy = ""
    
    if n <= 3:
        strategy = "uniform_fallback"
        final_indices = list(range(n))
    else:
        diffs = [0]
        for i in range(1, n):
            gray1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diffs.append(np.sum(cv2.absdiff(gray1, gray2)))
            
        peak_idx = int(np.argmax(diffs))
        
        candidate_indices = {
            0: "start",
            max(1, peak_idx - 1): "before_peak",
            peak_idx: "peak_motion",
            min(n - 2, peak_idx + 1): "after_peak",
            n - 1: "end"
        }
        
        final_indices = sorted(list(set([idx for idx in candidate_indices.keys() if 0 <= idx < n])))
        if len(final_indices) < 3:
            final_indices = sorted(list(set([0, n // 2, n - 1])))
            strategy = "uniform_fallback"
        else:
            strategy = "motion_peak_5"
            
    return [(idx, timestamps[idx]) for idx in final_indices], strategy

def create_dummy_video(path, num_frames=10, motion_at=5):
    fourcc = cv2.VideoWriter_fourcc(*'vp80')
    out = cv2.VideoWriter(path, fourcc, 10.0, (128, 128))
    for i in range(num_frames):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        if i == motion_at:
            # Huge motion spike
            frame = np.ones((128, 128, 3), dtype=np.uint8) * 255
        out.write(frame)
    out.release()
    with open(path, "rb") as f:
        return f.read()

class TestVLMSampling(unittest.TestCase):
    
    def test_short_recording(self):
        b = create_dummy_video("test_short.webm", 2, 1)
        res, strategy = sample_video_frames(b)
        self.assertEqual(strategy, "uniform_fallback")
        self.assertEqual(len(res), 2)
        
    def test_missing_video(self):
        res, strategy = sample_video_frames(None)
        self.assertEqual(strategy, "missing_video")
        self.assertEqual(len(res), 0)
        
    def test_normal_motion_peak(self):
        b = create_dummy_video("test_norm.webm", 15, 7)
        res, strategy = sample_video_frames(b)
        self.assertEqual(strategy, "motion_peak_5")
        indices = [r[0] for r in res]
        # Should include start (0), around peak (6, 7, 8), and end (14)
        self.assertIn(0, indices)
        self.assertIn(6, indices)
        self.assertIn(7, indices)
        self.assertIn(8, indices)
        self.assertIn(14, indices)
        
    def test_item_visible_only_one_frame(self):
        # A flash of an item (motion spike on exactly one frame)
        # It's at index 2
        b = create_dummy_video("test_flash.webm", 5, 2)
        res, strategy = sample_video_frames(b)
        self.assertEqual(strategy, "motion_peak_5")
        indices = [r[0] for r in res]
        self.assertIn(2, indices) # Must capture the flash!

if __name__ == "__main__":
    unittest.main()
