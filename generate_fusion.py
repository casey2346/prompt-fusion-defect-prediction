import pandas as pd

# Load static code metrics and LLM embeddings
X_metrics = pd.read_csv("X_metrics.csv")
X_llm = pd.read_csv("X_llm.csv")

# Concatenate features to create a fusion representation
X_fusion = pd.concat([X_metrics, X_llm], axis=1)
X_fusion.to_csv("X_fusion.csv", index=False)

print("✅ X_fusion.csv has been successfully generated (Fusion of static features + CodeBERT embeddings)")
