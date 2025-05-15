import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# Load the actual code snippets
df = pd.read_csv("code_snippets.csv")
texts = [f"What does the function do? The function is: {code}" for code in df["code"]]

# ✅ Clean input: retain only valid strings
texts = [str(t) for t in texts if isinstance(t, str) and not pd.isnull(t)]

# ✅ Align y.csv label rows with the number of valid texts
y = pd.read_csv("y.csv")["label"]
y = y.iloc[:len(texts)]
y.to_csv("y.csv", index=False)  # Overwrite with truncated labels

# Load the CodeBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        attention = inputs["attention_mask"].unsqueeze(-1)
        embedding = (last_hidden * attention).sum(1) / attention.sum(1)  # Mean pooling
        return embedding.squeeze().numpy()

# Generate embeddings for all prompt-formatted inputs
all_embeddings = [get_embedding(text) for text in texts]
X_llm = pd.DataFrame(all_embeddings)

X_llm.to_csv("X_llm_prompt.csv", index=False)
print("✅ X_llm_prompt.csv has been successfully generated (prompt-augmented CodeBERT embeddings)")
