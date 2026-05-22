# Intel Mac Setup

This folder contains a self-contained script for running the LIS pipeline on an Intel-based macOS machine. The script avoids the known numpy/torch/transformers compatibility issues on Intel Macs by pinning exact package versions.

## One-time environment setup

```bash
conda create -n lis_mac python=3.10 -y
conda activate lis_mac
pip install numpy==1.24.4 pandas==2.0.3 pyarrow==14.0.1 \
            torch==2.1.0 sentence-transformers==2.2.2 scikit-learn tqdm
```

## Convert PKL to Parquet

```bash
conda activate lis_mac
python run_on_mac.py --pkl Dimensions_LIS_1975_2024.pkl --mode convert
python run_on_mac.py --pkl OpenAlex_LIS_1975_2024.pkl   --mode convert
```

Output: `Dimensions_LIS_1975_2024.parquet` and `OpenAlex_LIS_1975_2024.parquet` in the same directory as the PKL files.

## Run SBERT encoding (optional)

```bash
conda activate lis_mac
python run_on_mac.py --pkl OpenAlex_LIS_1975_2024.pkl --mode sbert
```

- Encodes all abstracts with `all-MiniLM-L6-v2` into 384-dimensional vectors.
- Saves checkpoints every 5,000 texts — safe to interrupt and resume.
- Output: `sbert_output/OpenAlex_LIS_1975_2024_embeddings.npy`

If you run out of memory, reduce the batch size:
```bash
python run_on_mac.py --pkl OpenAlex_LIS_1975_2024.pkl --mode sbert --batch_size 64
```
