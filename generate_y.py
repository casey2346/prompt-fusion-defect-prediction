import numpy as np
import pandas as pd

# Generate 1000 binary labels (0 or 1)
y = np.random.randint(0, 2, size=1000)

# Save to CSV
pd.DataFrame({"label": y}).to_csv("y.csv", index=False)

print("✅ y.csv has been successfully generated with 1000 binary labels")
