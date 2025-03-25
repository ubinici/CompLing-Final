# Abkhaz POS Tagging with XLM-RoBERTa and Interactive Feedback

**Ümit Altar Binici**

---

## 1. Introduction

This project presents a part-of-speech (POS) tagging system for Abkhaz, a Northwest Caucasian language with rich morphology and minimal computational resources. Despite its complex linguistic structure—featuring polypersonal agreement, agglutinative forms, and a large consonant inventory—Abkhaz remains largely unexplored in natural language processing (NLP). The lack of annotated data and baseline tools has limited its visibility in the field and contributes to broader challenges around digital inclusion and language preservation.

To address this, we train a POS tagger on the Universal Dependencies (UD) Abkhaz Treebank using `xlm-roberta-large`, a transformer-based multilingual model pretrained on over 2TB of CommonCrawl data in 100 languages. XLM-RoBERTa is well-suited to low-resource settings due to its strong cross-lingual generalization capabilities, allowing us to fine-tune it on fewer than 2,000 annotated sentences while achieving competitive performance.

Our aim is twofold: first, to develop a reliable POS tagger for a low-resource language; and second, to demonstrate a sustainable workflow for expanding annotated data through human-in-the-loop feedback. An interactive demo was built using Streamlit to visualize predictions and collect corrections from users—native speakers or linguists—making the model more adaptive over time.

This project also stems from a personal academic motivation. Having previously worked with XLM-RoBERTa in a course assignment where I struggled to fully grasp its inner workings, this research gave me a chance to revisit the model more deeply and apply it meaningfully to an underrepresented linguistic context.

---

## 2. Methodology

### Dataset and Preprocessing

The dataset used in this study is the Abkhaz Universal Dependencies Treebank (`ab_abnc`), distributed in CoNLL-U format. It contains syntactically annotated sentences with Universal POS tags for each token. After parsing and shuffling, the dataset is split into training (80%), validation (10%), and test (10%) subsets to support robust evaluation.

Preprocessing involves tokenizing each sentence using the `XLMRobertaTokenizerFast` from Hugging Face, which performs subword tokenization while maintaining alignment with original word-level POS tags. Since one word may be split into multiple subword tokens, only the first subword is labeled, and the others are masked using `-100`. This strategy ensures that loss and evaluation are computed only over true token boundaries.

Universal POS tags are mapped to numerical class IDs, and the entire tokenized dataset is serialized using PyTorch for efficient reuse across training and evaluation scripts.

### Model Architecture

The POS tagging model is based on the `xlm-roberta-large` architecture. As a multilingual transformer pretrained on over 100 languages, it offers robust cross-lingual representations, particularly beneficial in low-resource settings. To prevent overfitting and reduce training time, the first 9 encoder layers along with the embedding layer are frozen during training.

A dropout layer with a rate of 0.3 and a fully connected classification head are appended to the transformer to produce per-token tag logits. Given contextualized embeddings from the transformer, the model computes a probability distribution over the POS tag set for each token position (excluding masked subwords).

### Model Selection

We initially experimented with the smaller `xlm-roberta-base` model due to its lower memory footprint and faster training times. However, its validation and test accuracies plateaued at approximately 80%, and improvements were limited even after hyperparameter tuning. Given the morphologically complex nature of Abkhaz, we hypothesized that the deeper representation capacity of `xlm-roberta-large` could yield better generalization. Subsequent experiments with the larger model confirmed this, resulting in validation accuracy exceeding 90% and improved F1 scores.

### Training Procedure

Model training is carried out using the AdamW optimizer with a learning rate of `2e-5`, weight decay of `0.01`, and gradient clipping (maximum norm `1.0`). A `ReduceLROnPlateau` scheduler monitors validation loss and reduces the learning rate if performance stagnates. The model is trained over multiple epochs, and the checkpoint with the best validation accuracy is retained for final evaluation.

Each batch contains tokenized input IDs, attention masks, and aligned POS labels. Only the first subword tokens are considered for loss computation, as indicated by the `-100` label mask. Training metrics, including loss and accuracy, are logged throughout.

### Evaluation and Feedback Loop

The model is evaluated on a held-out test set using token-level accuracy and macro-averaged F1 score. Subword-masked positions are excluded from metric calculations to ensure fair comparison. Misclassified tokens are logged and saved for future analysis.

To support iterative refinement, an interactive feedback system was implemented using Streamlit. Users can input Abkhaz sentences, review model predictions, and correct any errors via dropdowns. These user corrections are stored in a persistent log, enabling future supervised retraining. This human-in-the-loop setup encourages active engagement from native speakers or linguists, promoting both improved model performance and expanded annotated resources for Abkhaz.

--- 

## 3. Experiments & Ablation Studies

To better understand the contribution of different model components and training strategies, we conducted several ablation experiments. The goal was to evaluate not only raw performance but also training stability, generalization capacity, and implementation complexity—especially in the context of a low-resource language like Abkhaz.


### Final Model Configuration

The best-performing configuration utilized the `xlm-roberta-large` backbone with the first nine encoder layers frozen, followed by a dropout and a linear classification head. Subword tokens beyond the first position were masked during loss computation, ensuring that only one label per word contributed to training.

This configuration achieved strong results:

- **Validation Accuracy:** ~90.8%
- **Test Accuracy:** ~88.1%
- **Macro F1 Score (Test):** ~65.4%

This model was also the most stable during training and required minimal hyperparameter tuning after the initial setup.

### Initial Model Choice: XLM-RoBERTa-Base

Our initial experiments used `xlm-roberta-base` for its smaller size and faster training time. While this model trained efficiently, it consistently plateaued at around 80% validation accuracy and suffered from lower recall on less frequent POS tags. Given Abkhaz’s rich morphology, we hypothesized that the deeper and more expressive `xlm-roberta-large` model would better capture the underlying syntactic structures—an assumption validated by the improvement in both accuracy and F1 score.

### Attempted: CRF Decoding Layer

To capture dependencies between sequential tags, we experimented with adding a Conditional Random Field (CRF) layer on top of the classification head. While CRFs are often beneficial in sequence labeling tasks, integrating it with XLM-RoBERTa in this context proved problematic:

- **Training instability** was observed, particularly when handling the `-100` masked labels for subword tokens.
- **Runtime errors** and device-side assertions occasionally occurred during backpropagation.
- **Performance gains** were negligible or nonexistent, and validation F1 scores remained stagnant.

Due to these challenges and the added implementation complexity, the CRF layer was excluded from the final system.

### Attempted: Class Weighting in Loss

We also experimented with applying class weights to the loss function, derived from the frequency distribution of POS tags in the training data. This aimed to mitigate class imbalance and improve the recall of infrequent tags. However, class weighting led to:

- **Overcompensation**, causing the model to misclassify common tags as rare ones.
- **Gradient instability**, which occasionally slowed convergence or increased training loss volatility.

Ultimately, this technique did not improve the macro F1 score and was therefore removed from the final configuration.

### Summary of Findings

| Configuration                | Accuracy | F1 Score | Notes                                       |
|-----------------------------|----------|----------|---------------------------------------------|
| XLM-RoBERTa-Base            | ~80%     | ~55%     | Fast but underfit; weaker recall            |
| + CRF Layer                 | ~83–86%  | ~55–57%  | Unstable; little to no performance gain     |
| + Class Weights             | ~79–81%  | ~53–56%  | Overcompensated for rare tags               |
| **XLM-RoBERTa-Large (Final)** | **~88%** | **~65%** | Best generalization; no CRF or weights used |

These results suggest that, for clean and relatively small datasets like UD Abkhaz, simpler architectures with robust pretrained embeddings tend to generalize better than more complex decoding layers or reweighted losses.

---

## 4. Evaluation

The Abkhaz POS tagger was evaluated on a held-out test set using token-level accuracy and macro-averaged F1 score. Because the model uses subword tokenization, only the first subword of each word is labeled and evaluated, with all other positions masked using the `-100` index.

### Quantitative Metrics

The final model configuration—`xlm-roberta-large` with frozen lower layers and a linear classification head—achieved strong results on both validation and test sets:

| Metric              | Score    |
|---------------------|----------|
| **Validation Accuracy** | 90.8%   |
| **Test Accuracy**       | 88.1%   |
| **Macro F1 Score**      | 65.4%   |

While the accuracy figures indicate reliable overall performance, the F1 score reveals that the model is less consistent across infrequent or ambiguous POS tags—an expected limitation in low-resource, skewed-label distributions.

### Qualitative Error Analysis

Closer inspection of the model’s predictions highlighted several common misclassification patterns. Errors most frequently occurred between morphologically or functionally similar categories:

- **NOUN ↔ VERB**: Misclassification of nominalized verbs and verbal nouns.
- **ADV ↔ ADP**: Functional overlaps in adverbial modifiers and prepositional phrases.
- **AUX ↔ ADV**: Ambiguous auxiliary constructions misinterpreted as adverbs.

The following examples from the test set demonstrate typical model errors:

```
аз:     predicted = ADV     | true = ADP
амар:   predicted = NOUN    | true = INTJ
и:      predicted = VERB    | true = NOUN
Ақ:     predicted = NOUN    | true = VERB
А:      predicted = VERB    | true = NOUN
а:      predicted = AUX     | true = ADV
```

Some misclassifications—such as `:` being labeled as `ADP` or single-letter tokens like `А` and `а` being assigned syntactically implausible tags—suggest further structural issues, particularly in how the model processes low-frequency forms or punctuation.

### Tokenization-Related Challenges

A deeper source of error appears to be tied to the model’s reliance on XLM-RoBERTa’s default byte-level BPE tokenizer. This tokenizer was not optimized for Abkhaz and may split morphologically complex words in unnatural ways. Although we mask non-initial subwords during training and evaluation, the encoder’s contextual representations still incorporate these fragmented units, potentially introducing noise.

The misclassification frequency by tag type supports this hypothesis:

```
- NOUN (21)
- VERB (17)
- ADV (10)
- PRON (7)
- INTJ (4)
```

These are syntactically central categories often shaped by morphological suffixes or prefixes. If the tokenizer splits these incorrectly, it can obscure root meanings or grammatical cues. For example, a single Abkhaz verb form might encode subject, object, and tense features, but if subword splits bisect these morphemes, the model may misclassify the overall function of the token.

Despite masking during training, the first subword’s embedding remains influenced by adjacent subwords in the sequence. This means that even correctly labeled tokens may carry representational distortions introduced during tokenization.

### Implications and Opportunities

Tokenization thus emerges as both a foundational step and a critical bottleneck. Improving alignment between morphological structure and model input could substantially reduce the current error rate. Promising directions for future work include:

- Developing **custom Abkhaz-aware tokenizers** that respect affix boundaries or morpheme structure.
- Adopting **character-level encoders**, which avoid the pitfalls of byte-pair encoding altogether.
- Exploring **multi-subword aggregation** for POS prediction instead of relying solely on the first token’s representation.
- Using **unsupervised pretraining on Abkhaz corpora** to adapt existing models to the language’s segmentation patterns.

Such strategies may improve both interpretability and performance, particularly for categories most affected by segmentation errors.
---

## 5. Interactive Interface

To make the POS tagger accessible and support iterative refinement, we developed a lightweight interactive interface using [Streamlit](https://streamlit.io/). The interface serves both as a demonstration platform for Abkhaz POS tagging and as a feedback collection mechanism, allowing linguists and native speakers to directly engage with the system.

### Functional Overview

The Streamlit app enables users to input Abkhaz sentences and view model-predicted POS tags in a clean, color-coded table. For each token, the predicted tag is displayed, and users are invited to manually correct any incorrect labels using dropdown menus populated with the full POS tag set.

Corrected examples and user feedback are stored in a log file (`feedback_log.txt`), which mimics a simplified token–label format. These annotations are intended to be integrated into future model retraining workflows.

### Feedback Integration

To foster continuous learning, the app supports a basic human-in-the-loop loop:

1. **Prediction Phase**: User submits a sentence → model returns predicted tags.
2. **Correction Phase**: User confirms or modifies the tags.
3. **Logging Phase**: The corrected sample is saved locally.

Once a threshold number of feedback entries (e.g., 10) is reached, the user is prompted with the option to retrain the model. This retraining blends original training data with the new corrected samples, updating the model to reflect user guidance.

### Technical Details

The frontend leverages:

- **Tokenizer**: `xlm-roberta-large`, consistent with the training pipeline.
- **Model**: Final trained checkpoint (`pos_model.pth`) and optional feedback-tuned checkpoint.
- **Hardware**: CPU-only execution supported; GPU used during training.
- **Logging**: Feedback is appended to `feedback_log.txt` and parsed for retraining.

Example feedback record structure:

```
Input: Аҩы ҵы зҵааит
Predicted: [('Аҩы', 'NOUN'), ('шьҭы', 'VERB'), ('зҵааит', 'VERB')]
Feedback: No
Correction: ['PRON', 'ADV', 'VERB']
```

### Visualization and Insights

The app also includes a summary dashboard that visualizes the most commonly corrected tags using bar charts. This allows users and developers to quickly identify recurring weaknesses in the model and prioritize future improvements.

For instance, if a high number of corrections involve `ADV` being mislabeled as `ADP`, this pattern can inform future preprocessing or tokenizer adjustments.

### Impact

The integration of an interactive frontend transforms the POS tagger from a static model into a participatory linguistic tool. It enables:

- **Non-technical users** to interact with the model and contribute to annotation.
- **Linguists and native speakers** to provide valuable correction data.
- **Researchers** to monitor live model behavior and deploy updates easily.

This architecture not only supports practical model improvement but also opens the door to scalable, crowd-powered corpus expansion for under-resourced languages like Abkhaz.

---

## 6. Lessons Learned

This project offered several key insights into the design and deployment of POS taggers for low-resource, morphologically rich languages like Abkhaz.

- **Simpler models generalize better**: In the context of small, well-annotated datasets, complex additions such as CRF layers or class weighting did not provide consistent benefits. A straightforward linear classifier on top of `xlm-roberta-large` yielded the best performance.

- **Tokenizer behavior matters deeply**: Despite masking subwords during training, token segmentation still influenced final predictions. Suboptimal splits likely distorted morphological structure, especially for affixed or fused forms. The importance of tokenizer alignment with language-specific morphology became increasingly clear during error analysis.

- **User interaction is a powerful augmentation tool**: The integration of a feedback-based frontend allowed for scalable annotation and real-time model auditing. Even in its initial form, this human-in-the-loop component highlighted how expert corrections can guide future iterations and corpus expansion.

- **Metric selection is crucial**: While accuracy metrics remained high, the macro-averaged F1 score offered a more realistic view of the model’s weaknesses, particularly on rare or ambiguous tags.

---

## 7. Future Work

This project lays the groundwork for several promising research directions:

- **Custom Tokenization**: Replace or augment XLM-RoBERTa’s BPE tokenizer with one trained on Abkhaz-specific morpheme boundaries. Alternatively, explore character-level encoders to eliminate segmentation artifacts.

- **Semi-Supervised Learning**: Use unannotated Abkhaz corpora to pretrain or adapt language models before fine-tuning, increasing contextual familiarity and representation accuracy.

- **Sequence-Level Training Objectives**: Introduce label smoothing, multi-token alignment, or consistency regularization techniques to reduce prediction instability.

- **Corpus Expansion**: Leverage the Streamlit interface to crowdsource corrections from native speakers, then iteratively retrain the model using these enriched samples.

- **Cross-Lingual Transfer Analysis**: Examine how well the pretrained multilingual model leverages cross-lingual similarities, especially with related Caucasian languages, to inform zero-shot or few-shot scenarios.

---

## 8. Conclusion

In this study, we developed a part-of-speech tagging system for Abkhaz using `xlm-roberta-large`, addressing both the technical and linguistic challenges of low-resource NLP. Through careful preprocessing, model simplification, and iterative experimentation, we achieved high test accuracy (~88.1%) and meaningful macro F1 performance (~65.4%) on a dataset with fewer than 2,000 sentences.

Our analysis highlighted the importance of tokenizer behavior, particularly in morphologically rich languages, and revealed common POS confusions tied to subword segmentation. By integrating a Streamlit-based interactive interface, we extended the model into a dynamic tool for both demonstration and corpus enrichment—allowing users to participate directly in refining the system.

Ultimately, this project contributes not just a POS tagger, but a repeatable framework for extending NLP infrastructure in other underrepresented languages—built on pretrained models, user feedback, and careful error analysis.

---

## Acknowledgments

We thank the creators of the Universal Dependencies Abkhaz Treebank for making this valuable resource publicly available. Their contributions enabled the core training and evaluation stages of this project.

We also acknowledge the open-source communities behind Hugging Face Transformers and Streamlit. These tools allowed for fast prototyping, deployment, and visualization of the POS tagging system.

This project was partly inspired by earlier coursework, where XLM-RoBERTa was initially used. Revisiting the model in a more applied context enabled a deeper understanding and a more meaningful application toward underrepresented language technology.

Parts of this report were written and edited with the assistance of an AI language model to improve clarity, coherence, and structure.

--- 

## References

- **Universal Dependencies – Abkhaz Treebank**  
  https://universaldependencies.org/treebanks/ab_abnc/

- **XLM-RoBERTa Model Card (Facebook AI)**  
  https://huggingface.co/FacebookAI/xlm-roberta-large

- **Hugging Face Transformers Documentation – XLM-RoBERTa**  
  https://huggingface.co/docs/transformers/en/model_doc/xlm-roberta

- **Hugging Face Transformers GitHub Repository**  
  https://github.com/huggingface/transformers

- **Abkhaz Language – Wikipedia**  
  https://en.wikipedia.org/wiki/Abkhaz_language

- **Hewitt, B. G. (1979). Abkhaz: A Comprehensive Grammar**  
  Routledge (for linguistic background and grammar structure)

- **Meurer, P. (2009). A finite state approach to Abkhaz morphology and stress**  
  Retrieved from: https://clarino.uib.no/abnc/doc/Tbilisi2009-LNAI.pdf

- **Streamlit – The fastest way to build and share data apps**  
  https://streamlit.io/

- **Scikit-learn Documentation – F1 Score**  
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html

- **PyTorch Documentation – CrossEntropyLoss**  
  https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

- **Facebook AI – XLM-RoBERTa: Unsupervised Cross-lingual Representation Learning at Scale**  
  Conneau et al. (2020), *arXiv:1911.02116*  
  https://arxiv.org/abs/1911.02116

