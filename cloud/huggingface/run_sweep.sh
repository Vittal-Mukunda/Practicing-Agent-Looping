#!/bin/bash
set -e

cd /app/cadvae

# Download datasets using Kaggle API (secrets set in HF Space settings)
mkdir -p data/raw

echo "=== Downloading Dataset A (Customer Personality Analysis) ==="
kaggle datasets download -d iakash17/customer-personality-analysis -p data/raw --unzip

echo "=== Downloading Dataset B (eCommerce Oct) ==="
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store \
    -p data/raw -f 2019-Oct.csv --force
if [ -f data/raw/2019-Oct.csv.zip ]; then
    cd data/raw && unzip -o 2019-Oct.csv.zip && rm 2019-Oct.csv.zip && cd /app/cadvae
fi

echo "=== GPU check ==="
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

echo "=== Running Dataset A sweep ==="
python -m cadvae.eval.run_cadvae_sweep data=personality

echo "=== Running Dataset B sweep ==="
python -m cadvae.eval.run_cadvae_sweep data=ecommerce \
    eval.train_subsample=200000 model.max_epochs=40

# Copy results to persistent storage
echo "=== Copying results to persistent storage ==="
cp -r results/phase5 /data/results/ 2>/dev/null || true

echo "=== SWEEP COMPLETE ==="
ls -la results/phase5/
