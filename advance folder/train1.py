import re
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM

# ---- load fine-tuned model ----
model_path = "./festivals-ganpati-bert-mlm"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path).eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- load candidate names from your CSVs ----
festivals = pd.read_csv("festivals.csv")
ganpatis = pd.read_csv("ganpati.csv")

FESTIVAL_NAMES = sorted(set(festivals["Festival Name"].astype(str).str.strip()))
GANPATI_NAMES  = sorted(set(ganpatis["Ganpati Name"].astype(str).str.strip()))

def _find_mask_span(text: str):
    """
    Returns (prefix, suffix) around the first contiguous [MASK] block.
    Works for '[MASK]' or '[MASK] [MASK] ...'.
    """
    m = re.search(r'(\[MASK\](?:\s+\[MASK\])*)', text)
    if not m:
        raise ValueError("No [MASK] span found in the text.")
    return text[:m.start()], text[m.end():]

def _pll_score(sentence: str, span_text: str, prefix: str):
    """
    Pseudo log-likelihood score for the candidate span.
    Masks each tokenpiece of the span once and sums log probs.
    Length-normalized so longer names aren't penalized.
    """
    # token lengths without specials for prefix/span
    pref_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    span_ids = tokenizer(span_text, add_special_tokens=False)["input_ids"]

    enc = tokenizer(sentence, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    # start/end indices of span in the encoded sequence (with [CLS] at index 0)
    start = 1 + len(pref_ids)            # after [CLS] + prefix
    end   = start + len(span_ids) - 1

    tot_logprob = 0.0
    with torch.no_grad():
        for pos in range(start, end + 1):
            masked = input_ids.clone()
            masked[0, pos] = tokenizer.mask_token_id
            logits = model(input_ids=masked, attention_mask=attn).logits[0, pos]
            logprob = torch.log_softmax(logits, dim=-1)[input_ids[0, pos]]
            tot_logprob += float(logprob)

    return tot_logprob / max(1, len(span_ids))  # length-normalized

def rank_candidates(prompt: str, candidates, top_n=5):
    prefix, suffix = _find_mask_span(prompt)
    scored = []
    for cand in candidates:
        sent = (prefix + " " + cand + " " + suffix).replace("  ", " ").strip()
        score = _pll_score(sent, cand, prefix)
        scored.append((sent, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    print(f"\nInput: {prompt}")
    for i, (s, sc) in enumerate(scored[:top_n], 1):
        print(f"  {i}. {s}")

# ---- DEMO ----
rank_candidates("The most famous Ganpati in Mumbai is [MASK].", GANPATI_NAMES)
rank_candidates("People celebrate [MASK] with colors and joy.", FESTIVAL_NAMES)
rank_candidates("During [MASK], devotees worship Goddess Durga for nine nights.", FESTIVAL_NAMES)
rank_candidates("In Kerala, [MASK] is celebrated with boat races and feasts.", FESTIVAL_NAMES)
