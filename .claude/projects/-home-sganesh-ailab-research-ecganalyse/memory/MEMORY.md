# ECG-TransCovNet Project Memory

## Model Architecture
- Hybrid CNN-Transformer: CNN backbone (SKConv) + Transformer encoder-decoder with DETR object queries
- Input: [B, 7, 2400] (7 leads, 200Hz, 12s)
- CNN downsampling: 2400→600→150 tokens
- embed_dim=128, nhead=8, 3 encoder + 3 decoder layers, ~1.5M params

## Training Baseline (clean, balanced, 16k train / 3.2k val)
- Val accuracy: 87.4%, Macro F1: 0.877
- Strong: PVC, VFib, VTach, ST_Elevation, SVT, AFib, AFlutter, LBBB, RBBB (all F1>0.95)
- Weak: NORMAL_SINUS (F1=0.592) and AV_BLOCK_1 (F1=0.451) — mutual confusion due to morphological similarity
- HDF5 test: 12/13 correct (92.3%), only miss is Normal→AV_Block_1

## Key Files
- `train_ecg_transcovnet.py` — standalone training script
- `src/encoding/transcovnet.py` — per-lead encoder adapter for Phase 2
- `src/prediction/condition_classifier.py` — 16-class classifier for Phase 3/4
- `src/interpretation/rules.py` — neural augmentation of rule-based rhythm classification
- Model weights: `models/ecg_transcovnet/best_model.pt`

## Pipeline Integration
- FoundationModelAdapter supports model_type="transcovnet" (auto-detects weights)
- Phase 3 runs condition classification alongside fiducial extraction
- Rules engine accepts condition_prediction, overrides for confidence>=0.7
