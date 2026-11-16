# Audio-Visual Deepfake Detection

A working, end-to-end prototype that analyses a video's **audio** and **visual**
streams, derives numeric features, measures audio-visual synchronisation, and —
when a trained model is available — makes a real/fake prediction.

Upload a video in the Gradio app and it returns the extracted features, five
signal plots and a plain-language summary of what it measured.

---

## 1. What actually works (implemented)

| Module | Implemented functionality | File |
|---|---|---|
| 1. Input & preprocessing | Format/size/empty-file validation, OpenCV metadata probe, audio extraction with FFmpeg (system or the `imageio-ffmpeg` bundled binary), frame sampling at ~12 fps with a hard cap | `avdd/io_video.py` |
| 2. Audio features | MFCC (20 coefficients), zero-crossing rate, spectral centroid, roll-off, RMS energy, bandwidth, flatness → mean+std → **52 numbers** | `avdd/audio_feats.py` |
| 3. Visual / lip features | Face detection + 468 landmarks via **MediaPipe FaceLandmarker** (Tasks API), mouth opening, mouth aspect ratio, mouth width, lip-corner motion, normalised mouth-patch pixel motion → **16 numbers**; automatic fallback to an OpenCV Haar cascade (face box only) | `avdd/visual_feats.py` |
| 4. Audio-visual sync | Resamples both signals to a common grid, z-scores them, searches lag ±400 ms, reports zero-lag r, peak r and the best lag | `avdd/sync.py` |
| 5. Fusion + classifier | 68-dim fused vector (NaN for unmeasured values), median imputation fitted on the training split only, Random Forest, stratified split, accuracy / precision / recall / F1 / confusion matrix / ROC-AUC / 5-fold CV, bundle saved with its metrics | `avdd/fusion.py`, `avdd/model.py` |
| 6. UI + report | Gradio app: upload → analyse → prediction (or demo mode), feature table, five plots, plain-language explanation, `result.json` + `report.md` per video | `app.py`, `avdd/viz.py` |

Error handling is a first-class feature: a video without audio, without a
detectable face, corrupted, missing, or in an unsupported format produces a
clear message instead of a crash. Unmeasurable features stay `None`/`NaN` —
they are **never** filled with invented zeros.

---

## 2. Setup

### Option A — VS Code / terminal (used for this build)

Python 3.11 is recommended.

```bash
# 1. create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt
```

No separate FFmpeg install is needed: `imageio-ffmpeg` ships a binary.
If you already have FFmpeg on `PATH`, it is used instead.

The MediaPipe face-landmarker model (`models/face_landmarker.task`, ~3.6 MB) is
downloaded automatically on first use. To pre-fetch it:

```bash
python scripts/make_test_videos.py   # also warms up FFmpeg
```

### Option B — Google Colab

The code is plain Python and runs unchanged in Colab:

```python
!git clone <your-repo-url> avdd && %cd avdd
!pip install -q -r requirements.txt
!python scripts/make_test_videos.py
!python scripts/run_demo.py samples/sample_face_audio.mp4
```

For the UI in Colab, run the app and use the share link it prints
(`demo.launch(share=True)` — edit the last line of `app.py`, or run
`python app.py` after setting `GRADIO_SHARE=1`).

---

## 3. Run order (first time)

```bash
# 1. create the five synthetic test clips (not deepfakes - see the note in the script)
python scripts/make_test_videos.py

# 2. analyse one clip end to end, print the report
python scripts/run_demo.py samples/sample_face_audio.mp4

# 3. open the web interface
python app.py            # -> http://127.0.0.1:7860

# 4. (optional) run the test suite
python tests/test_pipeline.py
```

Every run writes to `outputs/<video_name>/`:

```
audio.wav                 extracted 16 kHz mono track
audio_overview.png        waveform, RMS envelope, spectral centroid
audio_spectrogram.png     mel spectrogram + MFCC heatmap
visual_mouth_motion.png   mouth opening / MAR / motion, grey = no face
sync_correlation.png      the two signals + cross-correlation vs lag
feature_vector.png        audio vs visual feature counts
result.json               every number and diagnostic, machine readable
report.md                 the plain-language explanation
```

Diagrams for the review:

```bash
python scripts/make_diagrams.py
# -> reports/architecture.png, reports/flowchart.png
```

---

## 4. Dataset and training

### Feature analysis mode (default)

No real/fake dataset ships with the repository, so the prototype extracts and
visualises the features without a real/fake label. Add labelled videos as shown
below to enable predictions.

### Adding a labelled dataset

```
data/
  real/   real_001.mp4  real_002.mp4  ...
  fake/   fake_001.mp4  fake_002.mp4  ...
```

Folder names are the labels; `genuine/` and `manipulated/` are accepted too.
Use clips of **similar length, resolution and content** per class, and keep
identity/production method disjoint between the train and test portions.

```bash
# extract features -> reports/features.csv, then train + evaluate
python scripts/train_model.py

# or, after a CSV already exists
python scripts/train_model.py --csv reports/features.csv --reuse-csv
```

Training is refused with a clear message if fewer than 10 usable videos or fewer
than 5 per class are found — a model that small would be meaningless.
`scripts/train_model.py` writes:

```
reports/features.csv         one row per video, all 68 features + diagnostics
reports/metrics.json         held-out metrics, confusion matrix, top features
models/av_deepfake_rf.joblib the bundle used by the pipeline and the UI
```

Splitting is done **per video**, never per frame, so frames from one clip cannot
appear in both train and test. The imputer is fitted on the training split only.

Once the bundle exists, the UI header switches from *feature analysis mode* to
*"Trained classifier loaded"* and shows the held-out metrics next to the
prediction. Predictions that needed imputation are flagged in the report.

**Suggested public datasets** (download manually — this project does not fetch
them): FakeAVCeleb, DFDC, KoDF, and the audio-visual subset of LAV-DF. Keep the
set small (a few hundred clips) so training stays CPU-friendly.

---

## 5. Tests

```bash
python tests/test_pipeline.py
```

| Case | Expectation |
|---|---|
| Video with audio + face | audio 52 features, face found, lips measured, sync computed |
| Video without audio | audio reported missing, visual still analysed, sync unavailable |
| Video with no face | visual unusable, lip features `None`, no crash |
| Corrupt file | `unreadable_video` error, no crash |
| Unsupported / missing file | `unsupported_format` / `missing_file` errors |
| CSV + training smoke test | features extracted, CSV written, training refuses <10 videos |

Verified result on this machine (Windows 11, Python 3.11.9, CPU):
`6/6 tests passed`. The clips are synthetic, so this proves the machinery works —
not that the classifier detects deepfakes.

---

## 6. Project layout

```
app.py                     Gradio interface (Module 6)
requirements.txt           pinned dependencies
avdd/
  config.py                every tunable number in one place
  io_video.py              Module 1 - validation, FFmpeg audio, frame streaming
  audio_feats.py           Module 2 - Librosa features
  visual_feats.py          Module 3 - MediaPipe/Haar face + lip features
  sync.py                  Module 4 - audio-visual cross-correlation
  fusion.py                Module 5a - 68-dim fused vector
  model.py                 Module 5b - Random Forest, metrics, bundle
  pipeline.py              end-to-end orchestration + explanation text
  viz.py                   all matplotlib figures
  dataset.py               labelled videos -> feature CSV
  _paths.py                Windows long-path workaround (see below)
scripts/
  make_test_videos.py      synthetic clips + error cases
  run_demo.py              command-line single-video demo
  train_model.py           feature extraction + training + evaluation
  make_diagrams.py         architecture + flowchart PNGs
tests/test_pipeline.py     the six test cases
reports/                   diagrams, metrics.json, features.csv
```

---

## 7. Known limitations (say these out loud in the review)

1. **No trained model ships with this repo.** Out of the box it is a feature
   analyser, not a detector. Accuracy claims require a labelled dataset.
2. **The sync score is a coarse proxy.** It correlates the audio RMS envelope
   with mouth opening at ~12 fps. Silence, music, head motion, occlusion, or a
   face that barely moves all depress the correlation for reasons unrelated to
   manipulation. It is a soft cue, never proof.
3. **Frame sampling is sparse** (~12 fps, max 240 frames, first 60 s only).
   Motion between analysed frames is not optical flow.
4. **Only one face is analysed** — the largest/first detection.
5. **Landmark quality depends on MediaPipe**; on the Haar fallback backend the
   mouth opening and MAR are `None` by design.
6. **Generalisation is unknown.** A Random Forest on hand-crafted features
   trained on one dataset will not transfer to unseen manipulation methods.
7. **Windows long-path note:** if the project lives in a very deep folder,
   numba (a Librosa dependency) fails with
   `DLL load failed while importing _box`. `avdd/_paths.py` rewrites long
   `sys.path` entries to their 8.3 short form automatically; this is a no-op on
   short paths and on macOS/Linux.

## 8. Future work (not implemented)

- CNN/LSTM or a pretrained visual encoder (e.g. EfficientNet/Xception) instead
  of hand-crafted lip geometry.
- Learned audio embeddings (wav2vec2 / Mel-spectrogram CNN) instead of MFCC
  statistics.
- Proper lip-reading or phoneme-viseme alignment for fine-grained AV sync.
- Temporal models over the full video instead of per-clip aggregates.
- Multi-face tracking, cross-dataset evaluation, calibration and threshold
  selection, and a proper ablation study.

See `reports/REVIEW2.md` for the module write-up, demo script and viva notes.
