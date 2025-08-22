# testing3.py
import re
import os
import time
import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM

import streamlit as st

# -------------------------
# Config — adjust if needed
# -------------------------
MODEL_PATH = "./festivals-ganpati-bert-mlm"   # your fine-tuned model folder
FESTIVALS_CSV = "festivals.csv"
GANPATI_CSV = "ganpati.csv"

# -------------------------
# Helpers: load model/data
# -------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForMaskedLM.from_pretrained(model_path).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device

@st.cache_data(show_spinner=False)
def load_candidates(festivals_csv, ganpati_csv):
    if not os.path.exists(festivals_csv) or not os.path.exists(ganpati_csv):
        raise FileNotFoundError("CSV files not found. Make sure festivals.csv and ganpati.csv exist.")
    festivals = pd.read_csv(festivals_csv)
    ganpatis  = pd.read_csv(ganpati_csv)
    FESTIVAL_NAMES = sorted(set(festivals["Festival Name"].astype(str).str.strip()))
    GANPATI_NAMES  = sorted(set(ganpatis["Ganpati Name"].astype(str).str.strip()))
    return FESTIVAL_NAMES, GANPATI_NAMES

# -------------------------
# Mask helpers & scoring
# -------------------------
def _find_mask_span(text: str):
    """
    Returns (prefix, suffix) around the first contiguous [MASK] block.
    Works for '[MASK]' or '[MASK] [MASK] ...'.
    Raises ValueError if no [MASK] found.
    """
    m = re.search(r'(\[MASK\](?:\s+\[MASK\])*)', text)
    if not m:
        raise ValueError("No [MASK] span found in the text. Add '[MASK]' where you want model to fill.")
    return text[:m.start()], text[m.end():]

def _pll_score(sentence: str, span_text: str, prefix: str, tokenizer, model, device):
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

# -------------------------
# Load model & data (cached)
# -------------------------
try:
    tokenizer, model, device = load_model_and_tokenizer(MODEL_PATH)
except Exception as e:
    st.stop()  # stop if model can't be loaded; Streamlit will display the exception below
try:
    FESTIVAL_NAMES, GANPATI_NAMES = load_candidates(FESTIVALS_CSV, GANPATI_CSV)
except Exception as e:
    st.error(str(e))
    st.stop()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Festival & Ganpati Prediction", layout="wide")
st.title("Festival & Ganpati Prediction (BERT MLM)")

default_prompts = [
    "The most famous Ganpati in Mumbai is [MASK].",
    "People celebrate [MASK] with colors and joy.",
    "During [MASK], devotees worship Goddess Durga for nine nights.",
    "In Kerala, [MASK] is celebrated with boat races and feasts.",
]

cols = st.columns([3, 1])
with cols[0]:
    prompt = st.text_input("Enter a sentence with [MASK]:", default_prompts[0])
with cols[1]:
    candidate_type = st.radio("Candidate set:", ["Ganpati Names", "Festival Names"])

top_n = st.slider("Top N predictions:", 1, 10, 5)
demo_choice = st.selectbox("Or choose a demo prompt:", ["None"] + default_prompts)
if demo_choice != "None":
    prompt = demo_choice

if st.button("Predict"):
    try:
        prefix, suffix = _find_mask_span(prompt)
    except ValueError as e:
        st.error(str(e))
    else:
        candidates = GANPATI_NAMES if candidate_type == "Ganpati Names" else FESTIVAL_NAMES

        # Warn if candidate list is huge
        if len(candidates) > 2000:
            st.warning(f"Candidate set has {len(candidates)} items — this can be slow. Consider reducing CSV size for faster results.")

        results = []
        start_time = time.time()
        with st.spinner("Scoring candidates (this may take a while)..."):
            for i, cand in enumerate(candidates, 1):
                sent = (prefix + " " + cand + " " + suffix).replace("  ", " ").strip()
                score = _pll_score(sent, cand, prefix, tokenizer, model, device)
                results.append((i, cand, sent, score))

        # sort by score desc
        results.sort(key=lambda x: x[3], reverse=True)
        end_time = time.time()

        # Show top-N as a nice table
        df = pd.DataFrame(results[:top_n], columns=["#","Candidate","Completed sentence","Score"])
        df["Score"] = df["Score"].map(lambda x: f"{x:.6f}")
        st.subheader(f"Top {top_n} predictions (computed in {end_time-start_time:.1f}s)")
        st.table(df)

        # Also show full top-k if user wants
        with st.expander("Show more (top 100)"):
            df_more = pd.DataFrame(results[:100], columns=["#","Candidate","Completed sentence","Score"])
            df_more["Score"] = df_more["Score"].map(lambda x: f"{x:.6f}")
            st.dataframe(df_more, use_container_width=True)

# Footer / quick notes
st.markdown("---")
st.markdown("**Notes:**\n- Make sure `festivals.csv` has a column `Festival Name` and `ganpati.csv` has `Ganpati Name`.\n- Run the app with `python3 -m streamlit run \"path/to/testing3.py\"`.")
