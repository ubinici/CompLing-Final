import torch
from torch import nn
from transformers import XLMRobertaModel


class POSModel(nn.Module):
    """
    POS tagger using XLM-RoBERTa with a linear classification head.
    """

    def __init__(self, num_tags, freeze_layers=9):
        super(POSModel, self).__init__()

        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")

        # Freeze embeddings and early transformer layers
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False
        for param in self.roberta.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return loss
        else:
            return logits  # not argmax!




