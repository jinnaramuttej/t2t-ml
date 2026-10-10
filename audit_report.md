# T2T Verification Pipeline Audit Report

## 1. Summary
- **Total Submissions:** 22
- **Correct Verdicts:** 20 / 22
- **False Approves:** 1 (Submission 07)
- **False Rejects:** 1 (Submission 20)
- **Acceptable Manual Reviews:** 2 (Submissions 16, 22)

## 2. Full Comparison Table

| ID | Timestamp | Actual Event (Manual Label) | Should | Model Verdict | Reject Reason | Scores (A, B, B_f1, B_f2, C) | Manual Flag | Match? |
|----|-----------|-----------------------------|--------|---------------|---------------|------------------------------|-------------|--------|
| 01 | 20261008_114801 | empty_hand_only | reject | reject | has_object | 0.002, 0.402, 0.408, 0.960, 1.0 | False | MATCH |
| 02 | 20261008_114829 | empty_hand_only | reject | reject | has_object | 0.000, 0.680, 0.207, 0.808, 1.0 | False | MATCH |
| 03 | 20261008_122712 | empty_hand_only | reject | reject | bin_present | 0.008, 0.001, 0.002, 0.007, 1.0 | False | MATCH |
| 04 | 20261008_123151 | empty_hand_only | reject | reject | has_object | 0.000, 0.988, 0.936, 0.910, 1.0 | False | MATCH |
| 05 | 20261008_123545 | wrong_bin / bin_not_visible | reject | reject | bin_present | 0.959, 0.034, 0.929, 0.572, 1.0 | False | MATCH |
| 06 | 20261008_123635 | genuine_disposal | approve | approve | | 0.991, 0.979, 0.961, 0.873, 1.0 | False | MATCH |
| 07 | 20261008_132235 | item_not_dropped (missed bin) | reject | approve | | 0.967, 0.838, 0.548, 0.637, 1.0 | False | FALSE APPROVE |
| 08 | 20261008_140713 | genuine_disposal | approve | approve | | 0.978, 0.998, 0.714, 0.202, 1.0 | False | MATCH |
| 09 | 20261008_141042 | genuine_disposal | approve | approve | | 0.948, 0.997, 0.716, 0.052, 1.0 | False | MATCH |
| 10 | 20261008_141404 | genuine_disposal | approve | approve | | 0.952, 0.948, 0.336, 0.168, 1.0 | False | MATCH |
| 11 | 20261008_141439 | genuine_disposal | approve | approve | | 0.923, 0.983, 0.083, 0.005, 1.0 | False | MATCH |
| 12 | 20261008_143335 | genuine_disposal | approve | approve | | 0.936, 0.975, 0.048, 0.521, 1.0 | False | MATCH |
| 13 | 20261008_143404 | genuine_disposal | approve | approve | | 0.953, 0.780, 0.320, 0.028, 1.0 | False | MATCH |
| 14 | 20261008_173126 | empty_hand_only | reject | reject | has_object | 0.000, 0.993, 0.018, 0.015, 1.0 | False | MATCH |
| 15 | 20261008_180532 | empty_hand_only | reject | reject | bin_present | 0.011, 0.001, 0.041, 0.029, 1.0 | False | MATCH |
| 16 | 20261008_181116 | photo_or_screen_of_trash | manual | manual | | 0.808, 0.391, 0.710, 0.771, 0.5 | True | ACCEPTABLE MANUAL_REVIEW |
| 17 | 20261008_182844 | wrong_bin / bin_not_visible | reject | reject | bin_present | 0.904, 0.201, 0.005, 0.037, 1.0 | False | MATCH |
| 18 | 20261008_182925 | wrong_bin / bin_not_visible | reject | reject | bin_present | 0.999, 0.115, 0.762, 0.060, 1.0 | False | MATCH |
| 19 | 20261008_182939 | hand_shaped_as_trash | reject | reject | bin_present | 0.973, 0.065, 0.611, 0.143, 1.0 | False | MATCH |
| 20 | 20261008_182957 | genuine_disposal | approve | reject | hand_empty | 0.987, 1.000, 0.914, 0.781, 0.2 | False | FALSE REJECT |
| 21 | 20261008_183042 | genuine_disposal | approve | approve | | 0.990, 1.000, 0.999, 0.855, 1.0 | False | MATCH |
| 22 | 20261009_183209 | genuine_disposal | approve | manual | | 0.988, 0.626, 0.975, 0.909, 1.0 | True | ACCEPTABLE MANUAL_REVIEW |

## 3. Analysis by Attack Type
- **Empty Hand Only (6 attempts):** Caught 6/6 by either `has_object` or `bin_present`. Pipeline is robust against this.
- **Wrong Bin / No Bin (3 attempts):** Caught 3/3 by `bin_present`.
- **Item Not Dropped / Missed Bin (1 attempt):** Caught 0/1. **Got through.** The pipeline cannot stop users who drop trash next to the bin, as it only checks for an empty hand and a bin in frame.
- **Hand Shaped as Trash (1 attempt):** Caught 1/1, but *only because the bin was missing*. The `has_object` check was fooled (A=0.973). The pipeline cannot currently stop this attack if performed over a real bin.
- **Photo/Screen of Trash (1 attempt):** Sent to manual review.

**Conclusion on Attacks:** The current pipeline cannot stop "item not dropped in bin" attacks, and it is highly vulnerable to "hand shaped as trash" if done over a valid bin. It also failed to catch reused identical scenes because the perceptual hash distance was too tight for live video jitter.

## 4. Per-Check Analysis & Thresholds
- **has_object (A):**
  - *Range:* Genuine disposals scored very high (0.93 - 0.99). Obvious fakes (empty hand) scored very low (0.00 - 0.01).
  - *Quality:* It discriminates well for empty hands, but completely failed on the "hand shaped as trash" attack (scored 0.973). 
  - *Threshold:* The current threshold (`LOW=0.35, HIGH=0.75`) is reasonable for separating the extremes, but the model logic itself needs improvement.
- **bin_present (B):**
  - *Range:* Genuine bins scored 0.62 - 1.00. Most non-bins scored very low (0.00 - 0.20), though outdoor structure trickery yielded 0.99 (id 14).
  - *Quality:* Good discriminator overall. The `HIGH=0.75` threshold safely pushes tricky outdoor bins (0.62) to manual review.
- **hand_empty (C):**
  - *Range:* Fakes and genuine alike easily score 1.000 because anyone can open their hand. Motion blur dropped a genuine submission to 0.200.
  - *Quality:* Practically **useless** for catching malicious actors (they all passed it), but it caused the *only* False Reject of a genuine user due to motion blur.
  - *Threshold:* 22 samples are too few, but the evidence strongly suggests relaxing or removing this strict mediapipe-based check.

## 5. Top 5 Concrete Fixes (by impact)
1. **Add a Video/VLM Check for the Drop Action:** The pipeline must verify the item physically enters the bin to stop "missed bin" and "kept item" attacks.
2. **Relax or Remove the MediaPipe `hand_empty` Check:** It catches no fakes (attackers easily open their palms) but causes false rejects for genuine users moving quickly (motion blur).
3. **Train against "Empty Fist" Attacks:** Update the `has_object` embeddings or model to distinguish between a hand actually holding an item versus an empty hand curled into a fist.
4. **Increase Hash Distance Threshold / Use Video Hashing:** A live camera hovering over the same bin produced bit distances of ~15. The threshold of 5 is too tight and allows reused videos to bypass deduplication.
5. **Implement Liveness / Screen-Replay Detection:** To prevent users from pointing their camera at a screen showing a valid disposal (as seen in the glitchy submission 16).

## 6. Recommended Next Test Round
- **Attacks to Add:** 
  - Drop the item outside the bin (more variations).
  - Use a string to pull the item back up.
  - Make a fist over a valid bin.
  - Replay a valid video from another screen.
- **Honest Disposals Needed:** At least 100-200 honest, varied disposals (different lighting, bins, motion blur, items) to confidently measure the False Reject rate, as the current sample (only ~10 genuine) is too small.
