# CentraleSupelec MLNS DSBA 2026

## Project Overview
This project tackles link prediction in an actor co-occurrence graph.  
Each node is an actor, edges represent co-occurrence on Wikipedia pages, and the objective is to predict missing links using both graph structure and processed node text features.

## Approach Summary
The pipeline in `code/train_submission.py` combines:
- Text pair features from `node_information.csv`
- Graph-topology pair features from positive train edges
- SVD embeddings from node text features
- Node2Vec embeddings from the train graph
- CatBoost + neural/linear predictors

Predictions are combined with rank-based ensembling into two submission variants:
1. `submission_crossrun_rank_meta_r2_w50_30_20_proba.csv` (stable blend)
2. `submission_crossrun_rank_meta_r2_aggr_w40_cat60_proba.csv` (more aggressive blend)

## Repository Contents
- Code: `code/train_submission.py`
- Report source: `report.tex`
- Original competition files:
  - `train.txt`
  - `test.txt`
  - `node_information.csv`
  - `random_predictions.csv`
  - `public_baseline.py`

## Reproducibility
Run from the project root:

```bash
python3 code/train_submission.py --max-runtime-minutes 15 --seed 42
```

The script writes and validates:
- `submission_crossrun_rank_meta_r2_w50_30_20_proba.csv`
- `submission_crossrun_rank_meta_r2_aggr_w40_cat60_proba.csv`

Kaggle submission format checks:
- Header: `ID,Predicted`
- Rows: `3498`
- IDs: `0..3497` in order

## Dependencies
- `numpy`
- `pandas`
- `scipy`
- `scikit-learn`
- `torch`
- `catboost`
