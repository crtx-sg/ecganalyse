# ECG Analysis Project Memory

## Project Structure
- `src/simulator/` - ECG simulator (16 conditions, 7 leads, 200Hz, 12s = 2400 samples)
- `train_ecg_transcovnet.py` - Standalone training script for ECG-TransCovNet model
- `ecg_transcovnet_notebook.py` - Original notebook code (5-class, 187-sample signals)
- `scripts/` - CLI tools for data generation
- `models/ecg_transcovnet/` - Trained model outputs

## ECG-TransCovNet Model
- Hybrid CNN (SK modules) + Transformer (DETR object queries), ~1.5M params
- CNN backbone: 2400 → 150 tokens (two stride-2 + pool stages)
- 16-class arrhythmia classification from simulator conditions
- Uses Focal Loss (alpha=0.5, gamma=2.0) for class imbalance
- Best val accuracy ~73% on first run (100 epochs, early stopped at 72)
- Strong on: VFib, ST_Elevation, VTach, SVT, AFib (F1 > 0.9)
- Weak on: AV_BLOCK_1 (too similar to normal), PAC, AV blocks (subtle differences)

## Key Technical Notes
- GPU: NVIDIA RTX 4000 Ada (20GB VRAM)
- Data generation: ~31s for 6400 samples, cached to .npz
- Training: ~1s/epoch on GPU with AMP
- Simulator generates 7 leads per call; model uses Lead II (ECG2) by default
- `--leads all` flag enables 7-channel input
