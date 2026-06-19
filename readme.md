# Land-Use Classification from Multispectral Satellite Imagery

A machine-learning pipeline that classifies Sentinel-2 satellite image patches (EuroSAT)
into ten land-use and land-cover categories, with a focus on measuring whether and how
information from non-RGB spectral bands improves classification.

The project trains a compact CNN from scratch on three input configurations (RGB-only, RGB
plus spectral indices, and a twelve-band multispectral set) under identical conditions, so
that the input bands are the only variable. A Random Forest on per-band statistics is added
as an explainability extension that corroborates the findings from a second model family.

Repository: https://github.com/AnikGhosh332/Land-Use-Classification

## Headline result

| Model | Channels | Test accuracy | Macro-F1 |
|-------|----------|---------------|----------|
| RGB baseline (CNN) | 3 | 94.5% | 0.944 |
| RGB + indices (CNN) | 6 | 95.9% | 0.957 |
| Multispectral (CNN) | 12 | 97.0% | 0.969 |

The full multispectral model improves on the RGB baseline by 2.5 percentage points. The
gain is modest but targeted, concentrating on the vegetation classes that look alike in RGB.
An independent Random Forest supports the result, with non-RGB features carrying roughly 75%
of total feature importance and adding 13 points of accuracy over an RGB-only feature set.

## Repository structure

```
.
├── main.ipynb                 # Main notebook (run this end to end)
├── main_old.ipynb             # Earlier working version, kept for reference
├── requirements.txt           # Python dependencies
├── data_index.csv             # Persisted patch index + 80/20 split (regenerated on run)
├── datasets/                  # Datasets (not included, see Setup below)
│   ├── EuroSAT_RGB/           # RGB JPGs, one folder per class
│   └── EuroSAT_MS/            # 13-band GeoTIFFs, one folder per class
├── artifacts/                 # Saved metrics, predictions, model weights (generated)
└── visualisations/            # Saved figures (generated)
```

The `artifacts/` and `visualisations/` folders are populated when the notebook runs. The
versions committed to the repository show the results from the reported run.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/AnikGhosh332/Land-Use-Classification.git
cd Land-Use-Classification
```

### 2. Create a virtual environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `rasterio` fails to install through pip on your platform, install it with conda instead,
which bundles the required GDAL libraries:

```bash
conda install -c conda-forge rasterio
```

### 4. Download the dataset

The datasets are not committed to the repository because of their size. Download both
versions of EuroSAT from the official source (https://github.com/phelber/eurosat) and
arrange them under a `datasets/` folder so that the structure matches the paths in the
notebook configuration.

```
datasets/
├── EuroSAT_RGB/        # RGB JPG version
│   ├── AnnualCrop/
│   ├── Forest/
│   └── ...             # one folder per class
└── EuroSAT_MS/         # 13-band multispectral (GeoTIFF) version
    ├── AnnualCrop/
    ├── Forest/
    └── ...             # one folder per class
```

Both versions share patch identifiers (for example `Forest_1.jpg` matches `Forest_1.tif`),
which the pipeline relies on to keep the RGB and multispectral models aligned.

If your folders are named differently, update `RGB_ROOT` and `MS_ROOT` in the configuration
cell at the top of `main.ipynb` to point at the correct paths.

## Running the project

Open the notebook and run it from top to bottom.

```bash
jupyter notebook main.ipynb
```

A clean, top-to-bottom run is important, because later cells depend on functions and
variables defined earlier, and the train/test split is created once and reused by every
model. To reproduce all results in one pass, choose "Restart Kernel and Run All Cells".

The notebook is organised into five sections:

1. Data preparation and preprocessing (indexing, exploratory analysis, the 80/20 split)
2. Feature extraction and representation learning (spectral indices and channel configurations)
3. Model training and evaluation (RGB baseline and two multispectral variants)
4. Analysis of model performance (objective and subjective RGB-versus-multispectral comparison)
5. Optional extension (Random Forest feature-importance explainability)

## Reproducibility

All randomness is controlled by a fixed seed (`RANDOM_SEED = 42`) set for NumPy, PyTorch,
and scikit-learn. The stratified 80/20 split is computed once, written to `data_index.csv`,
and read by every model, so the comparison is fair and repeatable. Per-channel normalisation
statistics are computed on the training data only, to avoid leakage into the test set.

### Reproducing the results from the command line

To run the entire pipeline non-interactively and regenerate all metrics, artifacts, and figures:

```bash
jupyter nbconvert --to notebook --execute --inplace main.ipynb
```

Expected runtime on CPU is roughly 20 to 60 minutes per CNN model depending on hardware,
and a few minutes for the Random Forest feature extraction (which is cached to
`artifacts/rf_features.csv` so re-runs are faster). Training is considerably quicker on a
CUDA GPU or Apple Silicon (MPS), both of which are detected automatically.

Note on exact reproducibility: on the same machine with a clean top-to-bottom run, results
are repeatable. Across different hardware (CPU versus GPU versus Apple MPS), small
floating-point differences can appear in the final decimals, which is normal for deep
learning and does not change the conclusions.


## Outputs

After a full run the following are produced:

- `artifacts/` contains per-model metrics (`*_metrics.json`), per-patch test predictions
  (`*_test_predictions.csv`), trained model weights (`*.pt`), the comparison tables, and the
  Random Forest results.
- `artifacts/` contains per-model metrics (`*_metrics.json`), per-patch test predictions (`*_test_predictions.csv`),    trained model weights (`*.pt`), the comparison tables (`comparison_summary.csv`, `per_class_deltas.csv`), and the Random Forest results (`rf_results.json`, `rf_feature_importance.csv`, `rf_features.csv`).
- `visualisations/` contains all figures, including the per-class F1 comparison, the
  confusion-difference matrix, sample predictions, and the Random Forest feature importance.

## Notes

- The CNNs are trained from scratch rather than fine-tuned from a pretrained network. This is
  a deliberate choice so that adding spectral bands is a clean, like-for-like change. Absolute
  accuracy is therefore slightly below a pretrained model, in exchange for a confound-free
  comparison.
- The full written report is provided separately as the project report document.