# Review 2 — Audio-Visual Deepfake Detection

Everything in this document is about **the prototype in this repository**. The
"implemented" list and the "planned" list are deliberately kept apart.

---

## 1. One-paragraph project summary

Deepfakes are videos whose face or voice has been synthesised or altered. Most
simple detectors look at only one stream. This prototype looks at both: it
extracts acoustic descriptors from the audio track, geometric descriptors from
the mouth and lips, and then checks whether the audio energy and the mouth
movement agree in time. A Random Forest combines the two feature sets into a
real/fake decision — but only if it has actually been trained on labelled
examples. Out of the box the system reports the measured features without a
prediction.

---

## 2. Module-by-module explanation

### Module 1 — Video input and preprocessing (`avdd/io_video.py`)
Validates the extension against a whitelist, rejects empty and oversized files,
opens the file with OpenCV and decodes one frame to prove it is readable, then
reads fps / frame count / resolution. The audio track is extracted with FFmpeg
to mono 16 kHz WAV (a system FFmpeg if present, otherwise the binary bundled in
`imageio-ffmpeg`). Frames are streamed in a single decode pass at about 12 fps,
capped at 240 frames and the first 60 seconds, so runtime stays bounded.
Every failure raises `VideoError` with a machine-readable code
(`unsupported_format`, `unreadable_video`, `missing_file`, …).

### Module 2 — Audio features (`avdd/audio_feats.py`)
Using Librosa on the extracted WAV: 20 MFCCs, zero-crossing rate, spectral
centroid, spectral roll-off, RMS energy, spectral bandwidth and spectral
flatness. Each descriptor is reduced to **mean and standard deviation** over
time, giving a fixed-length **52-number** vector regardless of clip length. The
frame-level RMS envelope is kept separately because Module 4 needs it.
Silent, empty and undecodable audio are detected and reported rather than
silently producing zeros.

### Module 3 — Visual and lip features (`avdd/visual_feats.py`)
Frames are downscaled to 480 px wide and passed to **MediaPipe FaceLandmarker**
(468 landmarks, Tasks API) — with an automatic fallback to an OpenCV Haar
cascade when MediaPipe is unavailable. From the lip landmarks we compute:

* mouth aspect ratio (outer lip opening ÷ mouth width),
* mouth opening normalised by face height,
* mouth width normalised by face width,
* lip-corner displacement per frame,
* frame-to-frame pixel change inside a fixed 64×32 mouth patch.

Each becomes a mean/std/p95 statistic plus the face detection rate →
**16 numbers**. On the Haar backend, opening/MAR are reported as `None` because
no landmarks exist; nothing is invented. Fewer than 5 detected faces marks the
visual block unusable.

### Module 4 — Audio-visual analysis (`avdd/sync.py`)
The RMS envelope and the mouth signal are resampled onto a common 25 Hz grid,
z-scored, and cross-correlated for lags of ±400 ms. The output is the zero-lag
correlation, the peak correlation and the lag at which it occurs, plus a plain
sentence describing what that means.

**Limitation, stated in the UI as well:** this is an energy-versus-motion
correlation, not true viseme/phoneme alignment. Music, silence, occlusion,
camera shake or a mostly still face all lower it without any manipulation. It is
one weak indicator among many.

### Module 5 — Fusion and classification (`avdd/fusion.py`, `avdd/model.py`)
Audio (52) and visual (16) features are concatenated into a **68-dimension**
vector; unmeasured values remain `NaN`. A `SimpleImputer(strategy="median")` is
fitted on the training split only and reused at inference, so there is no
leakage. The classifier is a Random Forest (`class_weight="balanced"`), trained
on a stratified split with the split performed **per video**, never per frame.
Reported metrics: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
(when both classes are present in the test split) and 5-fold CV on the training
split. `scripts/train_model.py` refuses to train on fewer than 10 videos or
fewer than 5 per class.

### Module 6 — Interface and reporting (`app.py`, `avdd/viz.py`)
Gradio UI: upload, choose backend, analyse. Shows the prediction **if a trained
bundle exists**, otherwise a clearly labelled feature-analysis banner. Displays
five plots (waveform + RMS + centroid, mel spectrogram + MFCCs, lip signal with
grey bands where no face was found, the sync cross-correlation, and the fused
feature counts), the full feature table, the raw run details as JSON, and a
plain-language explanation with limitations. Each run also writes `result.json`
and `report.md`.

---

## 3. Implemented vs planned

### Implemented and demonstrable today
- Format/size/corruption validation with specific error codes.
- FFmpeg audio extraction with a bundled binary (no manual install).
- Bounded frame sampling and single-pass decoding.
- 52 audio features (MFCC, ZCR, centroid, roll-off, RMS, bandwidth, flatness).
- MediaPipe landmark-based lip features (MAR, opening, width, corner motion,
  patch motion) with an automatic Haar fallback.
- Audio-visual cross-correlation with lag search and explicit caveats.
- 68-dim fused vector with NaN-preserving fusion.
- Random Forest baseline with leakage-free split, imputer, full metric set and a
  saved bundle that carries its own metrics.
- Gradio UI, five plots, JSON + Markdown reports.
- Six automated tests (happy path + all four failure paths + training smoke test).
- Architecture diagram and flowchart, generated by script.

### Planned / future work (NOT implemented)
- CNN or pretrained visual encoder (Xception/EfficientNet) instead of hand-crafted
  lip geometry.
- Learned audio representations (wav2vec2 or mel-spectrogram CNN) instead of MFCC
  statistics.
- Viseme/phoneme alignment or a lip-reading model for genuine AV sync checking.
- Temporal modelling (LSTM/Transformer) over the whole clip.
- Multi-face tracking, cross-dataset evaluation, probability calibration and
  threshold tuning, ablation study.

---

## 4. Results so far

| Claim | Status |
|---|---|
| The full pipeline runs end to end on CPU | **Verified** — `python tests/test_pipeline.py` → 6/6 pass |
| Audio and lip features are extracted from real video files | **Verified** — 52 audio + 16 visual features on the sample clip |
| Face detection works | **Verified** — MediaPipe 1.2.0 found a face in 75/75 sampled frames of the synthetic clip |
| Audio-visual sync is computed | **Verified** — cross-correlation reported with lag and r |
| Failure cases are handled | **Verified** — missing audio, no face, corrupt, unsupported, missing file |
| Training + inference code works | **Verified** on a throwaway synthetic set (CSV → bundle → prediction); the temporary model was deleted |

Accuracy numbers appear once `python scripts/train_model.py` has been run on a
labelled dataset — they come from that dataset's held-out split.

---

## 5. Demonstration script (about 4 minutes)

**Before the review:** run `python scripts/make_test_videos.py`, open
`python app.py` in one window, and have `reports/architecture.png` and
`reports/flowchart.png` open in another.

| Time | Action | What to say |
|---|---|---|
| 0:00 | Show `reports/architecture.png` | "Six modules: preprocessing, audio features, visual features, sync, fusion/classification, interface. Audio and video are analysed separately and then compared." |
| 0:30 | Show `reports/flowchart.png` | "Every stage can fail independently — no audio, no face, bad file — and each failure is reported instead of crashing." |
| 0:50 | In the UI, upload `samples/sample_face_audio.mp4`, press **Analyse video** | "Watch the feature table fill in and the plots appear." |
| 1:20 | Point at the plots | "Waveform and RMS energy; mel spectrogram and MFCCs; the lip signal — grey bands would mark frames without a face; the cross-correlation curve with the lag of the peak." |
| 2:00 | Upload `samples/sample_no_face_audio.mp4` | "No face detected, so the lip features stay empty rather than being filled with zeros, and sync is reported as unavailable." |
| 2:40 | Upload `samples/sample_face_no_audio.mp4` | "No audio track: the visual half still runs, and the report says exactly that." |
| 3:10 | Upload `samples/sample_corrupt.mp4` | "Corrupt file: a specific error code instead of a crash." |
| 3:30 | Close with training | "`scripts/train_model.py` builds the CSV, trains the Random Forest and reports a leakage-free confusion matrix on a labelled dataset." |

Backup if the UI misbehaves: `python scripts/run_demo.py samples/sample_face_audio.mp4`
prints the same report in the terminal.

---

## 6. Likely viva questions

**"Why Random Forest?"** Small tabular feature set, no long training, robust to
feature scaling, gives feature importances, and works with a few hundred
samples. A deep model would need far more data than a mini project can collect.

**"How do you avoid data leakage?"** The train/test split is over videos, not
frames; the median imputer is fitted only on the training split; the test set is
scored once, at the end.

**"Where does the accuracy number come from?"** From a held-out split over whole
videos, scored once after training, and it describes that dataset only.

**"How does the sync score work?"** It correlates the audio energy envelope with
mouth motion over a ±400 ms lag search and reports the zero-lag r, the peak r and
the lag of that peak.

**"Why does a missing face not break everything?"** Visual features stay `NaN`,
fusion keeps the NaN, and the imputer (fitted on training data) handles it.

**"What would you do with more time?"** Replace hand-crafted features with
learned CNN/wav2vec2 embeddings, add temporal modelling, and evaluate on a
public dataset such as FakeAVCeleb with a cross-dataset split.

---

## 7. Limitations list (for the slide)

1. No trained model ships with the repository, so the default output is feature analysis.
2. Sync is an energy/motion correlation, not viseme-level alignment.
3. ~12 fps sampling, first 60 s only, single face.
4. Hand-crafted features only — no learned representations.
5. Random Forest trained on one small dataset would not generalise.
6. MediaPipe landmark quality varies with pose, lighting and occlusion.
7. Threshold (0.5) is not calibrated.
