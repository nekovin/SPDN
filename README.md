# SPDN - Speckle Pattern Denoising Network

OCT image denoising framework for speckle separation.

## Requirements

```bash
pip install -e .
```

## Configuration

Before running, set your dataset paths in `configs/paths.yaml` or via environment variables:

```bash
export SDOCT_PATH=/path/to/Sparsity_SDOCT_DATASET_2012
export CHECKPOINT_DIR=./checkpoints
```

## Training

### SPDN Training

```bash
python scripts/spdn/train_spdn.py
```

### Baseline Training

```bash
python scripts/n2n/train_n2n.py
python scripts/n2s/train_n2s.py
python scripts/n2v/train_n2v.py
```

### N2-SPDN Training