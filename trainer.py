import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model import POSModel
from dataloader import load_data


def train_model(train_loader, val_loader, num_tags, epochs=10, lr=2e-5, max_grad_norm=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = POSModel(num_tags).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in tqdm(train_loader, desc=f"🟢 Epoch {epoch}/{epochs} Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        print(f"✅ Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        val_loss, val_acc = validate_model(model, val_loader, device)

        current_lr = scheduler.optimizer.param_groups[0]["lr"]
        print(f"📉 Learning Rate: {current_lr:.2e}")
        print(f"📈 Training Loss History: {loss_history}")

        scheduler.step(val_loss)

    model_path = "pos_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")
    return model_path


def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0
    total_correct = 0
    total_tokens = 0
    tag_errors = {}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="🔵 Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += loss.item()

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1)

            for pred_seq, label_seq, mask_seq in zip(preds, labels, attention_mask):
                for pred, true, m in zip(pred_seq.tolist(), label_seq.tolist(), mask_seq.tolist()):
                    if m == 1 and true != -100:
                        total_tokens += 1
                        if pred == true:
                            total_correct += 1
                        else:
                            tag_errors[true] = tag_errors.get(true, 0) + 1

    avg_loss = val_loss / len(val_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0

    print(f"✅ Validation Loss: {avg_loss:.4f}")
    print(f"✅ Validation Accuracy: {accuracy:.4f}")

    if tag_errors:
        top_errors = sorted(tag_errors.items(), key=lambda x: -x[1])[:5]
        print(f"🔍 Most Misclassified Tags: {top_errors}")

    return avg_loss, accuracy


if __name__ == "__main__":
    train_loader, val_loader, _, num_tags = load_data()
    train_model(train_loader, val_loader, num_tags)

