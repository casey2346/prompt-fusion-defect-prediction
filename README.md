# Fusing Static Metrics and Code Embeddings for Robust Software Defect Prediction

This repository contains code and data for the paper:

**Prompt-Optimized Fusion for Software Defect Prediction**

## 📄 Project Description

We propose a hybrid framework that combines:
- **Static code metrics** (e.g., LOC, complexity),
- with **semantic embeddings** generated from CodeBERT,
- optionally enhanced via **prompt-based transformation**.

We evaluate our approach using multiple variants and report improvements in F1 and AUC-ROC on synthetic datasets.

## 🧪 Folder Structure

```
📦 prompt-fusion-defect-prediction/
├── code_snippets.csv               # Code snippets for embedding input
├── y.csv                           # Binary classification labels
├── X_metrics.csv                   # Static handcrafted metrics (e.g., LOC, complexity)
├── X_llm.csv                       # Raw CodeBERT embeddings
├── X_llm_prompt.csv                # Prompt-enhanced CodeBERT embeddings
├── X_fusion.csv                    # Concatenation: metrics + raw embeddings
├── X_fusion_prompt.csv             # Concatenation: metrics + prompt embeddings
├── generate_data.py                # Generate synthetic code + metrics
├── generate_data_real.py           # Use real-world functions as input
├── generate_y.py                   # Create random binary labels
├── generate_fusion.py              # Concatenate metrics + embeddings
├── generate_fusion_prompt.py       # Concatenate metrics + prompt embeddings
├── generate_y_and_code_snippets.py # One-step synthetic generation
├── Result.py                       # Main script: run model and output results
```

## ⚙️ How to Run

1. (Optional) Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Generate embeddings and features:
   ```bash
   python generate_data.py
   python generate_y.py
   python generate_fusion.py
   ```

3. Run evaluation:
   ```bash
   python Result.py
   ```

## 🧠 Model

We use:
- **CodeBERT** (`microsoft/codebert-base`)
- **XGBoost** for classification with 5-fold stratified cross-validation

## 📈 Citation

If you use this code, please cite:

```
@article{rong2025promptfusion,
  title     = {Prompt-Optimized Fusion for Software Defect Prediction},
  author    = {Kexin Rong},
  journal   = {Journal of Systems and Software},
  year      = {2025}
}

```

## 📬 Contact

For any questions or issues, please contact: rongcasey@hotmail.com

