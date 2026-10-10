import unittest
from unittest.mock import patch, MagicMock
import os
import cv2
import numpy as np
from server import run_vlm_check, app

from fastapi.testclient import TestClient

client = TestClient(app)

class TestVLMCheckLogic(unittest.TestCase):
    
    @patch('server.requests.post')
    def test_vlm_responses(self, mock_post):
        # We'll pass dummy images so it bypasses video decoding
        img1 = b"fake_image_1"
        img2 = b"fake_image_2"
        
        # Test 1: IN_BIN
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"verdict": "IN_BIN", "reason_code": "ITEM_ENTERED_BIN", "note": "ok"}'}}]}
        mock_post.return_value = mock_resp
        res = run_vlm_check(None, img1, img2)
        self.assertEqual(res["vlm_verdict"], "IN_BIN")
        
        # Test 2: MISSED
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"verdict": "MISSED"}'}}]}
        res = run_vlm_check(None, img1, img2)
        self.assertEqual(res["vlm_verdict"], "MISSED")
        
        # Test 3: UNCERTAIN
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"verdict": "UNCERTAIN"}'}}]}
        res = run_vlm_check(None, img1, img2)
        self.assertEqual(res["vlm_verdict"], "UNCERTAIN")
        
        # Test 4: Malformed response
        mock_resp.json.return_value = {"choices": [{"message": {"content": 'just some text'}}]}
        res = run_vlm_check(None, img1, img2)
        self.assertEqual(res["vlm_verdict"], "UNCERTAIN")
        
        # Test 5: Exception / Timeout
        mock_post.side_effect = Exception("Timeout")
        res = run_vlm_check(None, img1, img2)
        self.assertEqual(res["vlm_verdict"], "ERROR")

    @patch('server.requests.post')
    def test_video_sampling_logic(self, mock_post):
        # Mock VLM to just return IN_BIN
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"verdict": "IN_BIN"}'}}]}
        mock_post.return_value = mock_resp
        
        # Create a synthetic video with 10 frames
        # Frames 0-4 are black, Frame 5 is white (peak motion), Frames 6-9 are black
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            temp_path = f.name
            
        out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'VP80'), 30, (128, 128))
        for i in range(10):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            if i == 5:
                frame.fill(255)
            out.write(frame)
        out.release()
        
        with open(temp_path, "rb") as f:
            v_bytes = f.read()
        os.remove(temp_path)
        
        res = run_vlm_check(v_bytes, b"img1", b"img2")
        
        # motion_peak_3 should have picked 3 frames around peak: 4 (before), 5 (peak), 6 (after)
        self.assertEqual(res["sampling_strategy"], "motion_peak_3")
        self.assertEqual(len(res["sampled_frame_timestamps"]), 3)
        # Check chronological order
        ts = res["sampled_frame_timestamps"]
        self.assertTrue(ts[0] <= ts[1] <= ts[2])
        
    @patch('server.requests.post')
    def test_video_fallback(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"choices": [{"message": {"content": '{"verdict": "IN_BIN"}'}}]}
        mock_post.return_value = mock_resp
        
        # Test missing / undecodable video fallback to 2 frames
        res = run_vlm_check(b"invalid_video_data", b"img1", b"img2")
        self.assertEqual(res["sampling_strategy"], "fallback_2_frames")

class TestValidateEndpoint(unittest.TestCase):
    
    @patch('server.run_vlm_check')
    @patch('server.decide')
    @patch('server.score_frames')
    def test_pipeline_verdicts(self, mock_score, mock_decide, mock_vlm):
        mock_score.return_value = (1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
        valid_jpg = cv2.imencode('.jpg', np.zeros((10,10,3), dtype=np.uint8))[1].tobytes()
        
        # Create dummy files for FastAPI
        files = {
            "frame1": ("f1.jpg", valid_jpg, "image/jpeg"),
            "frame2": ("f2.jpg", valid_jpg, "image/jpeg")
        }
        
        def run_test(initial_verdict, vlm_verdict, expected_final):
            mock_decide.return_value = (initial_verdict, "reason")
            mock_vlm.return_value = {"vlm_verdict": vlm_verdict}
            
            # Reset files pointer
            files["frame1"] = ("f1.jpg", valid_jpg, "image/jpeg")
            files["frame2"] = ("f2.jpg", valid_jpg, "image/jpeg")
            
            headers = {"x-api-token": "your_secure_api_token"}
            sess_resp = client.post("/session", headers=headers)
            sid = sess_resp.json()["session_id"]
            
            import server
            import time
            server.sessions[sid]["created_at"] -= 5  # Trick the age check
            
            response = client.post("/validate", data={"session_id": sid}, files=files, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["verdict"], expected_final)
            
        # If initial is reject, VLM is skipped, remains reject
        run_test("reject", "IN_BIN", "reject")
        run_test("manual_review", "IN_BIN", "manual_review")
        
        # If initial is approve
        run_test("approve", "IN_BIN", "approve")
        run_test("approve", "MISSED", "reject")
        run_test("approve", "UNCERTAIN", "manual_review")
        run_test("approve", "ERROR", "manual_review")
        run_test("approve", "MAYBE", "manual_review")

if __name__ == '__main__':
    unittest.main()
