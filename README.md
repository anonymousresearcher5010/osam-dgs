# OSAM-DGS: Operational, Structured and Annotated Multimodal German Sign Language Dataset for Machine Learning

Repository accompanying our NeurIPS 2026 dataset paper submission.  
This project provides code, schemas, and processing scripts to reconstruct dataset representations and run the experiments described in the paper.

## Data availability and licensing

This work builds on the publicly available DGS-Corpus Release 3, which is distributed under a restricted license for linguistic research.  
The original videos, annotations, and metadata are **not redistributed** in this repository and remain subject to the original [licensing terms](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/license_en.html).

Users must independently obtain access to DGS-Corpus and comply with its license.  
The code in this repository is released under the MIT License and does not include corpus data or derived corpus artifacts.  
Generated outputs (for example back-translations or pose estimations) remain subject to the original corpus licensing conditions.

Please cite both:
1. DGS-Corpus
2. This work

---

## Requirements

- Python `3.12.2` (managed with `pyenv`)
- Local environment for preprocessing
- GPU-capable environment for compute-heavy steps (recommended)
- For the experiments in the paper, we used a single NVIDIA A100 GPU for GPU-heavy stages

### Environment setup

`bash pyenv local 3.12.2 python -m venv .venv source .venv/bin/activate pip install --upgrade pip pip install -e .`


---

## End-to-end pipeline overview

| Stage | Script(s) | Output | GPU |
|---|---|---|---|
| 1. Data retrieval | `src/retrieve_DGS_Korpus_data.py`, `src/retrieve_DGS_Korpus_types.py` | Local raw/intermediate corpus representation | No |
| 2. Canonical dataset creation | `src/create_canonical_dataset.py` | Canonical dataset (schema-validated) | No |
| 3. Back-translation | `src/generate_backtranslations.py` | Canonical dataset with generated back-translations | Recommended |
| 4. Tabular views | `src/create_sentence_level_CSV.py`, `src/create_gloss_level_CSV.py` | Sentence-level and gloss-level CSV views | No |
| 5. Text-to-Gloss experiment | See section below | Model artifacts and evaluation outputs | Yes |
| 6. Pose generation (DWPose) | See section below | Pose cache files | Yes |
| 7. Pose-guided video generation | See section below | Generated signing videos | Yes |

---

## 1) Data retrieval

First retrieve DGS-Corpus Release 3 resources from the public source (transcripts, videos, pose data, metadata).

- Configure:
  - `configs/retrieving_data_config.yaml`
  - `configs/retrieving_types_config.yaml`
- Run:
  - `src/retrieve_DGS_Korpus_data.py`
  - `src/retrieve_DGS_Korpus_types.py`

## 2) Canonical dataset creation

Run:

- `src/create_canonical_dataset.py`

The generated canonical dataset is validated against:

- `configs/canonical_schema.yaml`

## 3) Back-translation

Run:

- `src/generate_backtranslations.py`

GPU acceleration is recommended for this step.

## 4) Tabular dataset views

Run:

- `src/create_sentence_level_CSV.py`
- `src/create_gloss_level_CSV.py`

These create tabular sentence-level and gloss-level views.

---

## 5) Text-to-Gloss experiment

### Prepare data (locally)

The text-to-gloss experiment uses the sentence-level CSV view.

1. Distill sentence-level data for train/eval:
   - `src/distill_sentence_glosses_CSV.py`

2. For RWTH-PHOENIX-Weather experiments used in the paper, download:
   - https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

3. Prepare the Phoenix dataset:
   - `prepare_phoenix_dataset.py`

### Training and evaluation (remote / GPU container)

Run:

- `src/text_to_gloss/run_finetune.sh`
- `src/text_to_gloss/run_finetune_phoenix.sh`

These scripts assume:
- a dedicated GPU-capable container/environment,
- required datasets mounted and available inside the container.

Outputs include evaluation `.txt` files and training artifacts.

---

## 6) Pose generation with DWPose (via MimicMotion)

We use DWPose through [MimicMotion](https://github.com/Tencent/MimicMotion).

### Prerequisites

- MimicMotion is installed and runnable according to its own instructions.
- GPU environment is available.
- Raw DGS data is available locally.

### Steps

1. Copy folder:
   - `src/pose_generation`
   into the **root** of the MimicMotion repository.
2. Provide the raw DGS-Corpus path to:
   - `pose_generation/compute_dwpose_cache.py`
3. Run pose cache generation inside the MimicMotion environment.

---

## 7) Pose-guided video generation (via MimicMotion)

We use [MimicMotion](https://github.com/Tencent/MimicMotion) for video synthesis.

### Prerequisites

- MimicMotion is installed and runnable according to its own instructions.
- GPU environment is available.

### Steps

1. Prepare experiment inputs in this repository:
   - `src/pose_guided_video_generation/prepare_video_experiment_data.py`
2. Generate MimicMotion-compatible YAML config:
   - `src/pose_guided_video_generation/generate_experiments_yaml.py`
3. Run MimicMotion inference using:
   - generated YAML config
   - prepared experiment data

### Important note for YAML generation

The YAML generator includes only experiment folders that contain both default files:
- `video.mp4`
- `reference_image.png`

If either file is missing, that folder is skipped.

---


## Troubleshooting

- **No output entries in generated experiment YAML**  
  Check each input subfolder contains both `video.mp4` and `reference_image.png`

- **Training script fails in local shell**  
  Use the intended GPU-enabled container/runtime.

- **Dataset retrieval fails**  
  Re-check DGS-Corpus access rights and retrieval config paths.

- **Pose generation fails**  
  Ensure `src/pose_generation` was copied to MimicMotion root and the raw data path is correctly set.

---

## Citation

Please cite:
1. DGS-Corpus
2. Our NeurIPS 2026 dataset paper (BibTeX entry will be added upon publication)

---

## Contact (anonymous during review)

For questions about the artifact, please contact: `anonymous.researcher5010@gmail.com`.