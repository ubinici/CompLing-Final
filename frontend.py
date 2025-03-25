import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizerFast
from model import POSModel
from preprocessor import id2tag
from trainer import train_model
from dataloader import load_data
from evaluator import evaluate_model
import os

# Load tokenizer and model (ensure consistency with trained model)
num_tags = len(id2tag)
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
model_path_base = "pos_model.pth"
model_path_feedback = "pos_model_feedback_v1.pth"
model = POSModel(num_tags)
model.load_state_dict(torch.load(model_path_base, map_location=torch.device("cpu")))
model.eval()

# Feedback log file
feedback_file = "feedback_log.txt"

# Annotate user input
def annotate_text(input_text):
    tokens = input_text.strip().split()
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    word_ids = encoded.word_ids()
    output = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != prev_word_idx:
            tag_id = predictions[idx]
            tag = id2tag.get(tag_id, "UNK")
            output.append((tokens[word_idx], tag))
            prev_word_idx = word_idx

    return output

# Streamlit Interface
st.set_page_config(page_title="Abkhaz POS Tagger", layout="centered")
st.title("📝 Abkhaz POS Tagging Demo")
st.write("Enter an Abkhaz sentence below to see predicted part-of-speech tags.")

# Model version toggle
model_version = st.radio("Select model version:", ["Base", "Fine-Tuned (with Feedback)"])
if model_version == "Fine-Tuned (with Feedback)" and os.path.exists(model_path_feedback):
    model.load_state_dict(torch.load(model_path_feedback, map_location=torch.device("cpu")))
    model.eval()
    st.success("✅ Feedback fine-tuned model loaded.")

user_input = st.text_input("Enter Abkhaz sentence:", "")

if user_input:
    annotations = annotate_text(user_input)
    st.markdown("### 🔍 Predicted POS Tags:")
    df = pd.DataFrame(annotations, columns=["Token", "Predicted Tag"])
    st.table(df)

    with st.expander("🧪 Technical Details"):
        st.markdown(f"""
        - **Tokenizer**: `xlm-roberta-large`  
        - **Model Architecture**: XLM-RoBERTa Large + Linear Classifier  
        - **Number of POS Tags**: `{num_tags}`  
        - **Device**: `{'CUDA' if torch.cuda.is_available() else 'CPU'}`  
        """)

    st.markdown("### 🧠 Feedback")
    feedback = st.radio("Were the POS tags correct?", ["Yes", "No"])
    corrected_tags = []

    if feedback == "No":
        st.markdown("#### ✏️ Provide corrections for incorrect tags:")
        tag_options = sorted(set(id2tag.values()))

        for idx, (token, predicted_tag) in enumerate(annotations):
            col1, col2 = st.columns([2, 2])
            with col1:
                st.markdown(f"**{token}** ({predicted_tag})")
            with col2:
                corrected = st.selectbox(
                    f"Correct tag for '{token}'",
                    options=tag_options,
                    index=tag_options.index(predicted_tag),
                    key=f"correction_{idx}"
                )
                corrected_tags.append(corrected)
    else:
        corrected_tags = [tag for _, tag in annotations]

    if st.button("Submit Feedback"):
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(f"Input: {user_input}\nPredicted: {annotations}\nFeedback: {feedback}\nCorrection: {corrected_tags}\n---\n")
        st.success("✅ Thank you for your feedback!")

# Feedback summary and visualization
try:
    with open(feedback_file, "r", encoding="utf-8") as f:
        feedback_entries = f.read().split("---\n")
        total_feedback = len([entry for entry in feedback_entries if "Input:" in entry])
        st.markdown(f"### 📊 Feedback Summary ({total_feedback} entries)")

        corrections = []
        for entry in feedback_entries:
            if "Correction:" in entry:
                try:
                    tags_line = entry.split("Correction:")[1].strip().split("\n")[0]
                    tags = eval(tags_line) if tags_line.startswith("[") else tags_line.split()
                    corrections.extend(tags)
                except Exception as e:
                    st.warning(f"⚠️ Error parsing correction: {e}")
                    continue

        if corrections:
            correction_counts = pd.Series(corrections).value_counts()

            with st.expander("📥 Feedback Insights"):
                st.markdown("Most frequently corrected POS tags:")

                fig, ax = plt.subplots(figsize=(8, 4))
                correction_counts.plot(kind="bar", ax=ax, color="skyblue")
                ax.set_title("Frequency of Corrected Tags")
                ax.set_xlabel("POS Tag")
                ax.set_ylabel("Count")
                ax.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        if total_feedback >= 10:
            st.warning("⚠️ Model has received over 10 feedback entries. You can retrain a feedback-augmented version.")
            if st.button("🔁 Retrain Feedback Model"):
                with st.spinner("Retraining with feedback data and base data blend..."):
                    train_loader, val_loader, test_loader, num_tags = load_data()
                    model_path = train_model(train_loader, val_loader, num_tags)
                    os.rename(model_path, model_path_feedback)
                st.success(f"✅ Feedback-tuned model saved as {model_path_feedback}")

        st.markdown("### 🧪 Evaluate Feedback-Tuned Model")
        if os.path.exists(model_path_feedback):
            if st.button("Run Evaluation on Test Set"):
                with st.spinner("Evaluating model..."):
                    test_loader = load_data()[2]  # get test loader only
                    evaluate_model(test_loader, num_tags, model_path_feedback)
except FileNotFoundError:
    pass
