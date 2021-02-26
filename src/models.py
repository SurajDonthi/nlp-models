import torch.nn as nn


class ClassifierModel(nn.Module):

    def __init__(self, embedding_size: int = 768, hidden_size: int = 128,
                 dropout: float = 0.3,
                 num_classes: int = 3):
        """
        [summary]

        Args:
            embedding_size (int, optional): [description]. Defaults to 768.
            hidden_size (int, optional): hidden_size of the . Defaults to 128.
            dropout (float, optional): Dropout is applited before the linear layers. \
                Set dropout to 0 if not required. Defaults to 0.3.
            num_classes (int, optional): [description]. Defaults to 2.
        """
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self._layers()

    def _layers(self):

        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_size, self.hidden_size),
            nn.ReLU(True),
            nn.Linear(self.hidden_size, self.num_classes),
            nn.Softmax()
        )

    def forward(self, embeddings):
        hidden_state_cls = embeddings[0][:, 0, :]
        logits = self.classifier(hidden_state_cls)

        return logits
