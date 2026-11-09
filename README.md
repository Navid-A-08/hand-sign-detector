# Hand Sign Detector

Recognizes **24 static ASL fingerspelling letters** (A–Y without J and Z) from a webcam or single images.
It uses the MediaPipe Hand Landmarker to get 21 hand landmarks, normalizes them into a 63-D vector, and classifies
that vector with a small MLP, with a random-forest baseline alongside.

> Status: v1 prototype, built and evaluated locally. J and Z need motion and are planned for v2.

## How it works

1. **Landmarks** (`src/hand_sign_detector/landmarks.py`): MediaPipe Tasks `HandLandmarker`. Images go in as captured
   (never mirrored). Coordinates are converted to pixel-aspect units, the wrist is moved to the origin, everything is
   divided by the wrist-to-middle-MCP distance, and hands labelled `Left` are mirrored into a canonical right hand.
   The features are **not** rotation-normalized, because H/U, G/Q and K/P differ mainly in orientation.
2. **Classifier** (`src/hand_sign_detector/models.py`): scikit-learn MLP (128-64) with a random-forest baseline.
3. **Inference** (`src/hand_sign_detector/inference.py`): `Uncertain` when the top probability is below 0.60,
   majority-vote smoothing over the last 10 frames, and a top-2 display. When no hand is found, the output is `nothing`.

## Setup

Python 3.11+ is required (NumPy 2.4 dropped 3.10).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py --model-only   # downloads models/hand_landmarker.task (~7.8 MB)
```

## Usage

```bash
python prepare_data.py            # download datasets + extract landmarks to data/landmarks/*.parquet (~10 min, 8 CPUs)
python train.py                   # source-disjoint CV + final models in models/
python evaluate.py                # held-out danrasband test -> reports/clean/
python predict_image.py img.jpg   # single image(s), JSON output; --annotate out_dir to save drawings
python webcam.py                  # live mode (q / Esc to quit)
python collect.py --signer navid  # record your own labelled landmarks (letter keys, space = rec, Esc = save)
pytest                            # unit tests
```

Recorded webcam sessions are saved in `data/landmarks/` with the same schema, so `train.py` picks them up automatically.

## Data

| Source | Used for | License | Samples kept (hand detected) |
|---|---|---|---|
| [danrasband ASL Alphabet Test](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) | **held-out test only** | CC0 | 750 images (683) |
| [Sau ASL Alphabet](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset), non-grassknoted parts | train | CC0 (as stated by the compiler) | 45,073 images (21,668) |
| [borisgraudt ASL Hand Landmarks](https://www.kaggle.com/datasets/borisgraudt/asl-alphabet-hand-landmarks) | train | CC0 | 9,703 landmark sets |
| [ASLNow!](https://huggingface.co/datasets/sid220/asl-now-fingerspelling) | train | MIT | 1,874 landmark sets |
| grassknoted ASL Alphabet (found inside Sau) | **excluded by default**, ablation only (`--include-gpl`) | GPL-2.0 | 75,000 images (55,369) |

Data hygiene in `prepare_data.py`:
- All 870 danrasband images are byte-identical copies inside Sau. They're dropped by name and hash.
- 709 more Sau images are full-resolution originals of danrasband test images. They're dropped with a perceptual
  near-duplicate check (32×32 thumbnail correlation).
- 71,970 Sau files are byte-identical copies of the GPL-2 grassknoted set under other names. They're attributed to
  grassknoted and excluded.

## Results

See `reports/` (JSON metrics, confusion matrices as PNG and CSV). Detection rate is always reported separately
from classification accuracy.

## Known limitations

- None of the sources has signer IDs, so the split is **source-disjoint**, not signer-disjoint. Recording more
  signers with `collect.py` is the highest-value next step.
- The landmark-only sources (borisgraudt, ASLNow!) don't record handedness or image size, so they can't be mirrored
  or aspect-corrected.
- The MLP is overconfident: most of its mistakes on the held-out set are still above the 0.60 threshold.
- There's no `nothing` training data from clean sources, so `nothing` currently means "no hand detected".
- `webcam.py` and `collect.py` have not been tested against a real camera yet.

## License

Free to use. You may use, copy, modify, and share this project for any purpose.

The datasets keep their own licenses (see the table above) and are not included in this repo.
