# Broadcast Audio Tagger

Labels every 2 seconds of a podcast or broadcast recording as **speech, music, speech over music, applause, laughter, noise or silence**, and returns a timeline:

```
0–8 s speech · 8–16 s music · 16–20 s applause · 20–28 s speech over music · 28–32 s silence
```

The project covers the full model lifecycle on a laptop (MacBook Air M3, 8 GB): data preparation and validation, experiment tracking, model comparison, a model registry with an automated promotion gate, serving through an API in Docker, and drift monitoring on live requests.

## Results

Test set: 422 clips, never used for training or model selection.

| Model | Labelled clips used | Test macro-F1 | Applause F1 | CPU ms per 2 s window* | Size |
|---|---|---|---|---|---|
| A. Small CNN on spectrograms, from scratch | 1,787 | 0.804 | 0.39 | 2.4 | 1.2 MB |
| B. CLAP zero-shot (prompt ensemble) | 0 | 0.859 | 0.88 | 74 | 113 MB |
| **C. Frozen CLAP embeddings + logistic regression** (champion) | 1,787 | **0.963** | **0.91** | 72 | 113 MB |

**Learning curve (model C, test macro-F1, mean of 5 random picks):**

| Labelled clips per class | 1 | 2 | 5 | 10 | 25 | 50 | all |
|---|---|---|---|---|---|---|---|
| Macro-F1 | 0.63 | 0.77 | 0.88 | 0.92 | 0.95 | 0.96 | 0.96 |

**Monitoring (simulated traffic built from test clips, model C served in Docker):**

| Scenario | Window accuracy | Drift monitor | What drifted |
|---|---|---|---|
| Normal (vs normal reference) | 0.98 | OK | nothing |
| Phone-quality audio (8 kHz, 300–3400 Hz, quieter, hiss) | **0.80** | **ALERT** | sample rate, loudness, silence share, low-confidence share |
| Music-heavy content (normal quality) | 0.96 | **ALERT** | predicted label mix only |

### Key findings
- **5 labelled clips per class beat zero-shot; 10 per class beat the CNN trained on all 1,787 clips.** Pretrained audio embeddings make new classes cheap to add.
- **Prompt wording matters:** averaging four descriptions per class raised zero-shot macro-F1 from 0.75 to 0.86.
- **Distillation helped data-rich classes but not applause.** Distillation transfers the teacher's knowledge through the student's training inputs, and the student saw only 70 applause clips; a larger transfer set would be the fix.
- **A promotion gate is only as good as its evaluation set:** the CNN passed the per-class F1 rule on validation (~24 applause clips) but scored 0.39 on test applause.
- **Phone audio cut accuracy from 0.96 to 0.80, mostly on music (recall 0.65) and speech over music (0.42),** because the phone band removes the frequencies that separate them. Mean confidence fell by only 0.04: the model was often confidently wrong, so input-feature drift, not confidence, raised the alert.
- **Drift is not always a performance problem.** A music-heavy content shift raised an alert while accuracy stayed at 0.96. Separating *input drift* (audio changed → check accuracy, maybe retrain) from *output drift* (content changed → investigate) points to the right response.

## How it works

```
MUSAN + ESC-50 ──► data pipeline (DVC) ──► 2,629 two-second clips, validated (Pandera)
                                              │
              ┌───────────────────────────────┼─────────────────────────────┐
              ▼                               ▼                             ▼
       A. CNN from scratch           CLAP embeddings (cached)          D. distilled CNN
                                      ├─► B. zero-shot (text prompts)
                                      └─► C. logistic regression
              └──────────────► MLflow tracking ◄────────────────────────────┘
                                      │
                    register (packaging check) ──► promotion gate ──► @champion
                                                                         │
                                         export ──► FastAPI + Docker ──► /tag, /health
                                                                         │
                                                         request log ──► Evidently drift report ──► MLflow
```

**Data.** MUSAN (speech, music, noise) and ESC-50 (applause, laughter, background sounds) are converted to 16 kHz mono, cut into 2 s windows, filtered for silence and loudness-normalised. Speech-over-music clips are made by mixing speech and music at 2–12 dB. Splits are by **recording**, so no recording appears in two splits; the pipeline found and fixed two ESC-50 source recordings that span official folds. Scripted checks verify every clip's format, loudness and labels, and DVC versions each dataset.

**Registry and promotion.** Each model is packaged with the same interface (2 s clips in, 6 class probabilities out) and registered in MLflow. A packaging check confirms the packaged model reproduces its training score. The gate promotes a version only if every class reaches F1 ≥ 0.80 on validation, it fits a 100 ms-per-window CPU budget, and it beats the current champion. Test scores are reported, never used to decide.

**Serving.** The API prepares audio exactly as in training, labels silence by a loudness rule, smooths predictions across neighbouring windows, merges them into segments, and logs each request (sample rate, loudness, silence share, confidence, label mix).

**Monitoring.** A reference is frozen from normal traffic. `drift_report.py` compares recent requests with it, feature by feature, using Evidently's statistical tests (Kolmogorov–Smirnov for numbers, chi-square / Z-test for categories), grouped into input features and model-behaviour features. It raises an alert if at least 30% of features drift (each test has a ~5% false-alarm rate; on unchanged data 0–2 of 13 features flag by chance) or if mean confidence drops by more than 0.05. Each report is saved as HTML and logged to MLflow.

## Project structure

```
configs/            settings for every step (data, models, registry, serving, monitoring)
src/data/           inventory, dataset build, validation
src/models/cnn.py   CNN architecture and log-mel features
src/                training (train_cnn, zero_shot, train_clap_head, train_distilled),
                    registry (register_models, promote, export_champion), tagger_pyfunc
src/serve/          FastAPI app and timeline logic
src/monitor/        traffic simulation, reference baseline, drift report
tests/              unit, API and monitoring tests
dvc.yaml            data pipeline: inventory → build → validate → CLAP embeddings
Dockerfile, docker-compose.yml
```

## Reproduce

```bash
# Environment
conda create -n audio-tagger python=3.11 -y && conda activate audio-tagger
conda install -c conda-forge ffmpeg -y
pip install -r requirements.txt

# Raw data (not included in this repository)
mkdir -p data/raw && cd data/raw
curl -L -o esc50.zip https://github.com/karoldvl/ESC-50/archive/master.zip && unzip -q esc50.zip && rm esc50.zip
curl -L -C - -o musan.tar.gz https://www.openslr.org/resources/17/musan.tar.gz && tar -xzf musan.tar.gz && rm musan.tar.gz
cd ../..

# Data pipeline and CLAP embeddings
dvc repro

# Models (all runs are logged to MLflow)
python src/train_cnn.py
python src/zero_shot.py
python src/train_clap_head.py
python src/train_distilled.py

# Registry
python src/register_models.py
python src/promote.py
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

# Serve
python src/export_champion.py
docker compose up --build
curl -F "file=@your_audio.wav" http://127.0.0.1:8000/tag

# Monitor (with the API running)
python src/monitor/simulate_traffic.py --scenario normal --n 40 --fresh
python src/monitor/make_reference.py
python src/monitor/simulate_traffic.py --scenario phone --n 30 --seed 3 --fresh
python src/monitor/drift_report.py
```

Run the tests with `python -m pytest -q tests`.

## Data and licences

Audio is not redistributed here. ESC-50 is available under CC BY-NC 3.0 and MUSAN from OpenSLR (resource 17).

## Next steps

- CI with GitHub Actions (tests, linting, Docker build on every push)
- Alert on strong drift in any single input feature, not only on the overall drift share
- Real-world evaluation on hand-labelled podcast clips, and retraining with phone-quality augmentation
