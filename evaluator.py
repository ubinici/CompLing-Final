import torch
from tqdm import tqdm
from sklearn.metrics import f1_score
from collections import defaultdict

from preprocessor import id2tag, tok
from model import POSModel
from dataloader import load_data


def evaluate_model(test_loader, num_tags, model_path="pos_model.pth"):
    """
    Evaluate model performance on the test set.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POSModel(num_tags).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    total_loss = 0
    total_correct = 0
    total_tokens = 0

    true_labels = []
    pred_labels = []
    incorrect = []
    tag_errors = defaultdict(int)

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="🔵 Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # === Mask out -100 labels ===
            logits_flat = logits.view(-1, num_tags)
            labels_flat = labels.view(-1)
            valid_mask = labels_flat != -100

            if valid_mask.sum().item() == 0:
                continue  # skip batches with no valid labels

            filtered_logits = logits_flat[valid_mask]
            filtered_labels = labels_flat[valid_mask]

            loss = loss_fn(filtered_logits, filtered_labels)
            total_loss += loss.item()

            # === Predictions ===
            predictions = torch.argmax(logits, dim=-1)

            for pred_seq, label_seq, mask_seq, input_seq in zip(
                predictions, labels, attention_mask, input_ids
            ):
                for i, (p, t, m) in enumerate(zip(pred_seq.tolist(), label_seq.tolist(), mask_seq.tolist())):
                    if m == 1 and t != -100:
                        total_tokens += 1
                        true_labels.append(t)
                        pred_labels.append(p)
                        if p == t:
                            total_correct += 1
                        else:
                            token_str = tok.convert_ids_to_tokens(input_seq[i].item()).replace("▁", "")
                            incorrect.append((token_str, id2tag.get(p, "UNK"), id2tag.get(t, "UNK")))
                            tag_errors[t] += 1

    # Convert for sklearn
    true_labels = [int(x) for x in true_labels]
    pred_labels = [int(x) for x in pred_labels]

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0
    f1 = f1_score(true_labels, pred_labels, average="macro")

    print(f"\n✅ Test Loss: {avg_loss:.4f}")
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    print(f"✅ Test F1 Score: {f1:.4f}")

    if tag_errors:
        top_errors = sorted(tag_errors.items(), key=lambda x: -x[1])[:5]
        print(f"\n🔍 Most Misclassified POS Tags: {[ (id2tag[k], v) for k, v in top_errors ]}")

    if incorrect:
        print("\n🔍 Sample Misclassified Tokens (Token, Predicted, True):")
        for tok_str, pred_tag, true_tag in incorrect[:10]:
            print(f"{tok_str}: {pred_tag} → {true_tag}")

    torch.save(incorrect, "misclassified_samples.pth")
    print(f"\n📁 Misclassified samples saved to misclassified_samples.pth")


if __name__ == "__main__":
    _, _, test_loader, num_tags = load_data()
    evaluate_model(test_loader, num_tags)


