python -m venv venv
. .\venv\Scripts\Activate.ps1
pip install torch torchvision open_clip_torch pillow imagehash opencv-python
python t2t_validator.py --selftest
python t2t_validator.py --frame1 a.jpg --frame2 b.jpg
python evaluate_clips.py --contact-sheet
python evaluate_clips.py
python evaluate_clips.py --sweep
