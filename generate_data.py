import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
X_metrics = np.random.rand(1000, 5)        # 5 handcrafted static code metrics
X_llm = np.random.rand(1000, 768)          # 768-dimensional LLM-based embeddings
y = np.random.randint(0, 2, size=1000)     # Binary defect labels

# Save to CSV files
pd.DataFrame(X_metrics, columns=[f'metric_{i}' for i in range(5)]).to_csv('X_metrics.csv', index=False)
pd.DataFrame(X_llm, columns=[f'dim_{i}' for i in range(768)]).to_csv('X_llm.csv', index=False)
pd.DataFrame(y, columns=['label']).to_csv('y.csv', index=False)

print("✅ All CSV files have been saved: X_metrics.csv, X_llm.csv, y.csv")
