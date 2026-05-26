# Retinal CVD Risk: Causal Mediation Analysis with Uncertainty Quantification

A reproducible four-stage research pipeline that predicts cardiovascular disease (CVD) risk
from retinal fundus images, explains *how* vascular features drive that risk through causal
mediation analysis, and provides coverage-guaranteed uncertainty intervals at every prediction.

This repository accompanies the paper:

> *Causal Mediation Analysis of Retinal Vascular Features and Cardiovascular Risk with
> Uncertainty Quantification: A Bayesian Conformal Prediction Framework.*

---

## Why this matters

Most deep learning models for retinal disease prediction are association machines — they
report a score but not a mechanism. This pipeline addresses two gaps:

1. **Causal transparency** — decomposes the effect of each vascular biomarker on disease risk
   into a *natural direct effect* (NDE) and a *natural indirect effect* (NIE) mediated through
   diabetic retinopathy (DR) and branch retinal vein occlusion (BRVO). This answers *how*
   retinal structure drives risk, not just *whether* it predicts it.
2. **Rigorous uncertainty quantification** — a Bayesian U-Net provides per-pixel epistemic
   uncertainty maps for segmentation; split-conformal prediction provides distribution-free,
   finite-sample-guaranteed risk intervals downstream.

### Key empirical findings

| Exposure biomarker | Mediator | Proportion mediated | NIE | 95% Bootstrap CI |
|--------------------|----------|--------------------:|-----|-----------------|
| Arteriolar tortuosity | DR | 33% | −0.51 | [−0.93, −0.09]* |
| Mean vessel width | DR | 24% | +0.65 | [+0.21, +1.10]* |
| Fractal dimension | DR | 21% | −0.55 | [−0.95, −0.14]* |
| Vessel density | BRVO | 5% | −0.10 | [−0.28, −0.03]* |

*95% bootstrap CI excludes zero (B = 1,000 resamples).*

Conformal risk regression: **89.6% empirical coverage** vs 90% nominal target (mean interval width 162.3).  
Conformal disease classification: **91.7% empirical coverage**, mean prediction set size 1.43.

---

## Pipeline overview

```
RFMiD fundus images (1,920 training images, 512×512 px)
        │
        ▼
[Stage 1]  Bayesian U-Net
           Mean-field variational inference · S = 20 MC forward passes
           → Binary vessel mask + per-pixel epistemic uncertainty map (σ̂ = 0.0827)
        │
        ▼
[Stage 2]  Vascular Biomarker Extraction
           Medial-axis skeletonisation · box-counting fractal dimension
           Arc-to-chord tortuosity · distance-transform vessel width
           → 10 features: ρ, A, D_f, κ, B, W, C, L, β, σ̂
        │
        ▼
[Stage 3]  Causal Mediation Analysis
           Two-model OLS · sequential ignorability · bootstrap CIs (B = 1,000)
           → NDE, NIE, ATE, proportion mediated for each exposure–mediator pair
        │
        ▼
[Stage 4]  Split-Conformal Prediction
           50/35/15 train/calibration/test split · α = 0.10 (90% nominal)
           → Coverage-guaranteed risk intervals  P(Y ∈ C(X)) ≥ 1 − α
        │
        ▼
Report: CVD risk score · causal pathways · uncertainty-quantified intervals
```

---

## Dataset: RFMiD

The pipeline uses the **Retinal Fundus Multi-disease Image Dataset (RFMiD)**:

- 3,200 fundus images with 45 binary disease labels per image (DR, BRVO, ARMD, …)
- `Disease_Risk` column used as the primary CVD proxy outcome
- No patient metadata in RFMiD — age, sex, and comorbidity variables are imputed from
  realistic population distributions for the causal models
- Available on Kaggle (CC-BY 4.0):
  `sohailaelsayed/retinal-fundus-multi-disease-rfmid`

```python
import kagglehub
path = kagglehub.dataset_download("sohailaelsayed/retinal-fundus-multi-disease-rfmid")
```

> This repository does not redistribute the dataset. The pipeline also runs end-to-end
> on purely synthetic data (`--synthetic`) without any download.

---

## Quick start

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
pip install kaggle kagglehub

# 2. Smoke-test on synthetic data (no dataset download needed)
python scripts/run_pipeline.py --config configs/default.yaml --synthetic

# 3. Run tests
pytest tests/

# 4. Full run (requires RFMiD — see Dataset section above)
python scripts/run_pipeline.py --config configs/default.yaml
```

**Preferred training environment**: Google Colab (T4 / A100 GPU).
See `notebooks/train_colab.ipynb` for the full Colab workflow including dataset download.

---

## Repository layout

```
retinal-cvd-causal-uq/
├── configs/
│   └── default.yaml              # single source of truth for all hyperparameters
├── src/
│   ├── pipeline.py               # orchestrates all four stages
│   ├── data/
│   │   ├── dataset.py            # RFMiD loader + synthetic data generator
│   │   └── labels.py             # Framingham-style CVD label synthesis
│   ├── models/
│   │   ├── bayesian_unet.py      # variational Bayesian U-Net (epistemic UQ)
│   │   ├── mc_dropout_unet.py    # MC-Dropout U-Net (faster approximate UQ)
│   │   └── factory.py            # build / train / inference helpers
│   ├── features/
│   │   └── extraction.py         # 10 vascular biomarkers
│   ├── causal/
│   │   ├── mediation.py          # NDE / NIE / proportion mediated
│   │   └── causal_forest.py      # heterogeneous treatment effects (CATE)
│   └── uncertainty/
│       └── conformal.py          # split-conformal prediction
├── scripts/
│   ├── run_pipeline.py           # CLI entry point (argparse)
│   └── generate_pipeline_figure.py
├── tests/
│   └── test_pipeline.py          # pytest — full pipeline on synthetic data < 1 min
├── notebooks/
│   ├── train_colab.ipynb         # primary Colab training notebook (GPU)
│   └── analysis.ipynb            # EDA and result exploration
└── outputs/                      # generated artifacts (gitignored)
```

---

## Technologies

| Category | Library / Tool | Version |
|----------|---------------|---------|
| Deep learning | [PyTorch](https://pytorch.org/) | ≥ 2.0 |
| Image processing | [scikit-image](https://scikit-image.org/) | ≥ 0.21 |
| Machine learning | [scikit-learn](https://scikit-learn.org/) | ≥ 1.3 |
| Statistical modelling | [statsmodels](https://www.statsmodels.org/) | ≥ 0.14 |
| Causal inference | [econml](https://github.com/microsoft/EconML) | ≥ 0.15 |
| Numerical computing | [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) | ≥ 1.24 / ≥ 1.10 |
| Data manipulation | [Pandas](https://pandas.pydata.org/) | ≥ 2.0 |
| Visualisation | [Matplotlib](https://matplotlib.org/) | ≥ 3.7 |
| Configuration | [PyYAML](https://pyyaml.org/) | ≥ 6.0 |
| Data acquisition | [kagglehub](https://github.com/Kaggle/kagglehub) | latest |
| Training hardware | NVIDIA T4 / A100 (Google Colab) | — |
| Language | Python | ≥ 3.9 |

---

## Reproducibility

- Random seed `42` set globally in `configs/default.yaml`
- All intermediate artifacts written to `outputs/` (`save_intermediate: true`)
- Synthetic dry-run: `python scripts/run_pipeline.py --synthetic` completes in < 2 min
- Full test suite: `pytest tests/` — no dataset required

---

## Citation

```bibtex
@article{asare2025retinalcvd,
  author  = {Asare, Blessing},
  title   = {Causal Mediation Analysis of Retinal Vascular Features and
             Cardiovascular Risk with Uncertainty Quantification:
             A Bayesian Conformal Prediction Framework},
  year    = {2025}
}
```

See also `CITATION.cff`.

---

## Contributors

| Name | Email | Role |
|------|-------|------|
| Blessing Asare | asareblessing8@gmail.com | Author and maintainer |

---

## License

MIT License (code). The RFMiD dataset is licensed separately under CC-BY 4.0 —
review the dataset license before redistribution.
