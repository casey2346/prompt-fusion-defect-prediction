import pandas as pd
import numpy as np

# Generate 1000 identical code snippets
code_snippets = ["def foo(): return 42"] * 1000
pd.DataFrame({"code": code_snippets}).to_csv("code_snippets.csv", index=False)
print("✅ code_snippets.csv has been successfully generated")

# Generate 1000 binary labels
labels = np.random.randint(0, 2, size=1000)
pd.DataFrame({"label": labels}).to_csv("y.csv", index=False)
print("✅ y.csv has been successfully generated")
