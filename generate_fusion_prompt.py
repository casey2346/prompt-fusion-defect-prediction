import pandas as pd

# Load static code metrics and prompt-based LLM embeddings
X_metrics = pd.read_csv("X_metrics.csv")
X_llm_prompt = pd.read_csv("X_llm_prompt.csv")

# Concatenate features to create a fused representation
X_fusion_prompt = pd.concat([X_metrics, X_llm_prompt], axis=1)

# Save the fused features to a CSV file
X_fusion_prompt.to_csv("X_fusion_prompt.csv", index=False)
print("✅ X_fusion_prompt.csv has been successfully generated (Fusion of static features + prompt-based CodeBERT embeddings)")
