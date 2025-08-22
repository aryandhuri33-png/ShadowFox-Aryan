import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, DataCollatorForLanguageModeling
from torch.optim import AdamW
from tqdm import tqdm

# ---------------- Read Datasets ----------------
festivals_df = pd.read_csv("festivals.csv", encoding="utf-8")
ganpati_df = pd.read_csv("ganpati.csv", encoding="utf-8")

print("Festivals dataset shape:", festivals_df.shape)
print("Ganpati dataset shape:", ganpati_df.shape)

# ---------------- Combine Text ----------------
festivals_df["combined_text"] = (
    festivals_df["Festival Name"].astype(str) + " - " +
    festivals_df["Festival Information"].astype(str)
)

ganpati_df["combined_text"] = (
    ganpati_df["Ganpati Name"].astype(str) + " - " +
    ganpati_df["Ganpati Information"].astype(str)
)

# ---------------- Clean Text ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

festivals_df["combined_text"] = festivals_df["combined_text"].apply(clean_text)
ganpati_df["combined_text"] = ganpati_df["combined_text"].apply(clean_text)

# Remove empty rows
festivals_df = festivals_df[festivals_df["combined_text"] != ""]
ganpati_df = ganpati_df[ganpati_df["combined_text"] != ""]

# ---------------- Create Corpus ----------------
corpus = pd.concat(
    [festivals_df["combined_text"], ganpati_df["combined_text"]],
    ignore_index=True
).tolist()

print("Total combined texts:", len(corpus))
print("\nSample entries:")
for i in range(5):
    print("-", corpus[i])

# ---------------- Tokenizer ----------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ---------------- Dataset Class ----------------
class FestivalGanpatiDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=max_length)

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

dataset = FestivalGanpatiDataset(corpus, tokenizer)

# ---------------- Train/Val Split ----------------
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# ---------------- Data Collator ----------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=data_collator)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=data_collator)

# ---------------- Model ----------------
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- Optimizer ----------------
optimizer = AdamW(model.parameters(), lr=5e-5)

# ---------------- Training ----------------
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# ---------------- Save Model ----------------
model.save_pretrained("./festivals-ganpati-bert-mlm")
tokenizer.save_pretrained("./festivals-ganpati-bert-mlm")
print("✅ Model fine-tuning complete and saved to ./festivals-ganpati-bert-mlm")
