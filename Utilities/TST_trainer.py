import os
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim

from torch.utils.data import DataLoader, TensorDataset
from aeon.datasets import load_classification

def load_dataset(name: str, batch_size: int = 4):
    """
    Loads and preprocesses the dataset.
    Returns train and test DataLoaders.
    """
    X_train, y_train = load_classification(name, split="train")
    X_test, y_test = load_classification(name, split="test")

    X_train = _preprocess_series(X_train)
    X_test = _preprocess_series(X_test)

    y_train, y_test = _remap_labels(y_train, y_test)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def _preprocess_series(X: np.ndarray) -> torch.Tensor:
    """
    Swap to shape (N, seq_len, channels) and convert to float tensor
    """
    arr = X.astype(np.float32)
    arr = np.swapaxes(arr, 1, 2)
    return torch.tensor(arr)


def _remap_labels(y_train: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Convert to zero-based int64 tensors
    """
    t_train = torch.tensor(y_train.astype(np.int64))
    t_test = torch.tensor(y_test.astype(np.int64))
    min_val = int(t_train.min())
    if min_val == 0:
        t_train -= min_val
        t_test  -= min_val
    return t_train, t_test


class TimeSeriesTransformer(nn.Module):
    """
    A Transformer-based classifier for multivariate time series
    """
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        seq_len: int,
        d_model: int = 128,
        n_head: int = 8,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # convolutional embedding
        self.conv1 = nn.Conv1d(input_dim, d_model//4, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(d_model//4)
        self.conv2 = nn.Conv1d(d_model//4, d_model//2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(d_model//2)
        self.conv3 = nn.Conv1d(d_model//2, d_model, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(d_model)

        # positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # classification head
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, channels)
        x = x.transpose(1, 2)  # (B, channels, seq_len)
        x = torch.relu(self.bn1(self.conv1(x)))  # conv1 -> bn1 -> relu
        x = torch.relu(self.bn2(self.conv2(x)))  # conv2 -> bn2 -> relu
        x = torch.relu(self.bn3(self.conv3(x)))  # conv3 -> bn3 -> relu
        x = x.transpose(1, 2)  # (B, seq_len, d_model)
        x = x + self.pos_enc   # add positional encoding
        x = self.transformer_encoder(x)    # transformer encoder
        x = x.transpose(1, 2)  # (B, d_model, seq_len)
        x = self.pool(x).squeeze(-1)  # global pooling -> (B, d_model)
        return self.classifier(x)     # (B, num_classes)


class Trainer:
    """
    Encapsulates training and evaluation loops
    """
    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            device: torch.device = None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optim.RAdam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()


    def train(self, epochs: int = 100):
        self.model.train()
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(Xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * Xb.size(0)
            avg_loss = total_loss / len(self.train_loader)
            lr = self.optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:03d}/{epochs} -- Loss: {avg_loss:.4f} -- LR: {lr:.6f}')

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for Xb, yb in self.test_loader:
            Xb, yb = Xb.to(self.device), yb.to(self.device)
            preds = self.model(Xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        acc = correct / total
        print(f'Accuracy: {acc * 100:.2f}%')
        return acc


def main():
    # hyperparameters
    dataset_name = "JapaneseVowels" # simply specify name from https://www.timeseriesclassification.com/dataset.php
    batch_size = 4
    num_epochs = 100
    model_path = f"TST_{dataset_name.lower()}.pth"

    # load data
    train_loader, test_loader = load_dataset(dataset_name, batch_size)

    # determine num_classes
    train_labels = train_loader.dataset.tensors[1]
    test_labels  = test_loader.dataset.tensors[1]
    num_classes  = int(torch.cat([train_labels, test_labels]).max().item()) + 1

    # model instantiation
    sample_seq, _ = next(iter(train_loader))
    seq_len, channels = sample_seq.shape[1], sample_seq.shape[2]
    model = TimeSeriesTransformer(
        input_dim=channels,
        num_classes=num_classes,
        seq_len=seq_len,
    )

    trainer = Trainer(model, train_loader, test_loader)

    # load or train
    if os.path.exists(model_path):
        trainer.model.load_state_dict(torch.load(model_path, map_location=trainer.device))
        print(f'Model loaded from {model_path}')
    else:
        trainer.train(epochs=num_epochs)
        torch.save(trainer.model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    trainer.evaluate()


if __name__ == '__main__':
    main()
    