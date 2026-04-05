from torch import nn

from .modeling_scbert import ScBertModel


class ClassificationHead(nn.Module):
    def __init__(self, seq_len: int, dropout: float, hidden_dim: int, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(seq_len, 512)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, hidden_dim)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        return self.fc3(x)


class ScBertClassifier(nn.Module):
    def __init__(self, backbone: ScBertModel, architecture: dict, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = ClassificationHead(
            seq_len=architecture["max_seq_len"],
            dropout=architecture.get("dropout", 0.0),
            hidden_dim=architecture.get("head_hidden_dim", 128),
            num_classes=num_classes,
        )

    def forward(self, input_ids):
        output = self.backbone(input_ids=input_ids, return_dict=True)
        return self.classifier(output.last_hidden_state)
