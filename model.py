import torch
from torch import nn
from transformers import XLMRobertaModel


class POSModel(nn.Module):
    """
    POS tagger using XLM-RoBERTa with a linear classification head.
    """

    def __init__(self, num_tags, freeze_layers=9):
        """
        Initializes the POSModel.

        Args:
            num_tags (int): Number of unique POS tags.
            freeze_layers (int): Number of bottom transformer layers to freeze.
        """
        super(POSModel, self).__init__()

        # Load pretrained XLM-RoBERTa (large) model
        self.roberta = XLMRobertaModel.from_pretrained("xlm-roberta-large")

        # Freeze embedding layer (word + position + token type embeddings)
        for param in self.roberta.embeddings.parameters():
            param.requires_grad = False

        # Freeze first `freeze_layers` transformer layers to retain general knowledge
        for param in self.roberta.encoder.layer[:freeze_layers].parameters():
            param.requires_grad = False

        # Dropout for regularization after contextualized representations
        self.dropout = nn.Dropout(0.3)

        # Linear classifier maps hidden states to POS tag logits
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs [batch_size, seq_len]
            attention_mask (torch.Tensor): Attention mask [batch_size, seq_len]
            labels (torch.Tensor, optional): Gold POS tags [batch_size, seq_len]

        Returns:
            If labels are provided:
                torch.Tensor: Cross-entropy loss
            Else:
                torch.Tensor: Raw logits [batch_size, seq_len, num_tags]
        """
        # Run inputs through the XLM-RoBERTa model
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last hidden state (contextual token representations)
        hidden_states = outputs.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]

        # Apply dropout and classification head
        logits = self.classifier(self.dropout(hidden_states))  # shape: [batch_size, seq_len, num_tags]

        # If gold labels are provided, return the loss (training mode)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding/special tokens
            loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))  # flatten for loss computation
            return loss
        else:
            # During inference, return raw logits (argmax is applied externally)
            return logits





