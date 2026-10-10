import os, time
import server
server.scorer = server.ClipScorer()

labels = {
    't2t-1.mp4': 'reject',
    't2t-2.mp4': 'approve',
    't2t-3.mp4': 'reject',
    't2t-4.mp4': 'manual_review',
    't2t-5.mp4': 'reject',
    't2t-6.mp4': 'approve',
    't2t-7.mp4': 'manual_review'
}

for i in range(1, 8):
    clip = f't2t-{i}.mp4'
    v_file = f'clips/{clip}'
    with open(v_file, 'rb') as f: v_bytes = f.read()
    
    import cv2
    cap = cv2.VideoCapture(v_file)
    ret, frame1 = cap.read()
    frames = [frame1]
    while True:
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
    cap.release()
    frame2 = frames[-1]
    
    from PIL import Image
    img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
    
    A, B, C, _, _, _ = server.score_frames(img1, img2, 0.5, 0.5)
    init_verdict, _ = server.decide(A, B, C)
    
    vlm_verdict = 'SKIPPED'
    ts = []
    final_verdict = init_verdict
    
    if init_verdict == 'approve':
        res = server.run_vlm_check(v_bytes, b'', b'')
        vlm_verdict = res.get('vlm_verdict', 'UNCERTAIN')
        ts = res.get('sampled_frame_timestamps', [])
        
        if vlm_verdict == 'MISSED':
            final_verdict = 'reject'
        elif vlm_verdict == 'UNCERTAIN':
            final_verdict = 'manual_review'
        elif vlm_verdict == 'ERROR':
            final_verdict = 'manual_review'
        elif vlm_verdict != 'IN_BIN':
            final_verdict = 'manual_review'
            
    points = 'YES' if final_verdict == 'approve' else 'NO'
    
    print(f'Clip: {clip} | Human: {labels.get(clip, "?")} | CLIP: {init_verdict} | TS: {[round(t,1) for t in ts]} | VLM: {vlm_verdict} | Final: {final_verdict} | Points: {points}')
    time.sleep(2)
