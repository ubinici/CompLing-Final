# Abkhaz POS Tagger with XLM-RoBERTa + Interactive Feedback

This project implements a **Part-of-Speech tagging system for Abkhaz**, a morphologically rich and low-resource language. Built on top of **XLM-RoBERTa**, it supports not only fine-tuning and evaluation, but also **live user correction and feedback-based retraining** via a **Streamlit frontend**.

The system is designed with low-resource NLP in mind—supporting **token alignment**, **subword masking**, and **human-in-the-loop learning**.


## Author
**Ümit Altar Binici**


## Setup Instructions

### Required: Download the UD Abkhaz Treebank
To run the preprocessing and training scripts, you must first download the Universal Dependencies Abkhaz dataset:

- Download link: [https://raw.githubusercontent.com/UniversalDependencies/UD_Abkhaz-ABNC/master/ab_abnc-ud-train.conllu](https://raw.githubusercontent.com/UniversalDependencies/UD_Abkhaz-ABNC/master/ab_abnc-ud-train.conllu)

Once downloaded, place the file in the root directory of this project and ensure the filename matches what the script expects:

**Expected location:** `./ab_abnc-ud-train.conllu`

---

## Directory Structure

```plaintext
|  
├── main.py               # Full training & evaluation pipeline
├── preprocessor.py       # Prepares dataset: tokenization + label alignment
├── dataloader.py         # Loads preprocessed data into PyTorch DataLoaders
├── model.py              # POS tagger model (XLM-RoBERTa + classifier)
├── trainer.py            # Training logic with validation + LR scheduling
├── evaluator.py          # Evaluation on the test set
├── frontend.py           # Streamlit app with live tagging, feedback, retraining
├── tokenized_abkhaz_dataset.pth   # Preprocessed dataset
├── feedback_log.txt      # Collected feedback (uploaded to Hugging Face)
├── pos_model.pth  # Model checkpoint (downloaded from Hugging Face)
├── README.md             # This file
```

## Requirements

- Python 3.11+
- PyTorch >= 2.0.0
- Transformers >= 4.34.0
- Streamlit, Datasets, Pandas, Matplotlib
- `huggingface_hub` for model syncing

## Features

### Transformer-based Tagging
- Uses `xlm-roberta-large`
- Freezes bottom 9 layers for efficiency
- Applies subword masking and alignment

### Interactive Feedback Loop
- Users can correct predictions live
- Corrections are logged and used for retraining

### Evaluation Pipeline
- Accuracy, macro F1, misclassification patterns
- Confusion across tag pairs analyzed

## How to Run

### 1. Preprocessing Dataset

```bash
python preprocessor.py
```
Outputs: `tokenized_abkhaz_dataset.pth`

### 2. Train + Evaluate Model (Full Pipeline)

```bash
python main.py
```

- Ensures dataset is preprocessed
- Trains model and evaluates it on test set
- Outputs: `pos_model.pth`, evaluation logs

### 3. Launch Streamlit Demo App (for tagging & feedback)

```bash
streamlit run frontend.py
```

The app allows real-time tagging and lets users correct errors. Feedback is stored as `feedback_log.txt` and used for fine-tuning.

## Feedback Integration
- Every submission logs corrected POS tags.
- Once threshold is reached, user can trigger retraining.
- Updated model is pushed back to Hugging Face and pulled live by the app.

## Learning Outcomes
- Fine-tuning multilingual Transformers for morphologically rich languages
- End-to-end data processing, training, and evaluation with PyTorch
- Interactive NLP system with user-in-the-loop learning
- Deployment pipeline from model training to continuous feedback loop

## Acknowledgements
- [Universal Dependencies: Abkhaz Treebank](https://universaldependencies.org/treebanks/ab_abnc/index.html)
- Hugging Face, Streamlit, PyTorch, Transformers community

