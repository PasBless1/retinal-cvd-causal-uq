#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# mBRSET acquisition helper.
#
# mBRSET is distributed under its own license and is NOT redistributed by this
# repository. This script documents where to obtain it and verifies the
# expected on-disk layout afterwards.
# ---------------------------------------------------------------------------
set -euo pipefail

DATA_DIR="${1:-data/mbrset}"

cat <<'EOF'
========================================================================
mBRSET (Mobile Brazilian Retinal Dataset)
========================================================================
5,164 fundus images / 1,291 patients, with clinical metadata
(age, sex, diabetes duration, comorbidities, treatments).

Dataset paper:
  https://www.nature.com/articles/s41597-025-04627-3

To download:
  1. Open the dataset paper above.
  2. Follow the "Data Records" / "Data Availability" link to the hosting
     repository (PhysioNet or Harvard Dataverse).
  3. Create an account / accept the data use agreement if required.
  4. Download and extract the archive.

Expected layout after extraction (relative to repo root):

  data/mbrset/
  ├── images/                # all fundus .jpg/.png files
  └── metadata.csv           # one row per image/patient

If the metadata column names differ from the defaults, edit
configs/default.yaml -> data.columns to match.
========================================================================
EOF

mkdir -p "${DATA_DIR}/images"

if [[ -f "${DATA_DIR}/metadata.csv" ]]; then
  n_img=$(find "${DATA_DIR}/images" -type f \( -name '*.jpg' -o -name '*.png' \) | wc -l)
  echo "[ok] metadata.csv found; ${n_img} image files detected."
else
  echo "[pending] Place metadata.csv and images under ${DATA_DIR}/ then re-run."
  echo "          Until then, use: python scripts/run_pipeline.py --synthetic"
fi
