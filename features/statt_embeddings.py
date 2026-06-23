import numpy as np
import torch
import torch.nn as nn
from utils.config import CNN_EMBEDDING_DIM, BATCH_SIZE, LEARNING_RATE, CNN_EPOCHS
from utils.logger import get_logger

logger = get_logger("STAtt")


class SpatialAttention(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.fc = nn.Linear(n_channels, n_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.fc(x.mean(dim=-1)), dim=-1)
        return x * weights.unsqueeze(-1)


class TemporalAttention(nn.Module):
    def __init__(self, n_time: int):
        super().__init__()
        self.fc = nn.Linear(n_time, n_time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.fc(x.mean(dim=1)), dim=-1)
        return x * weights.unsqueeze(1)


class STAttEmbedder(nn.Module):
    """
    Spatial-Temporal Attention embedding module.
    Input: (batch, n_channels, n_time)
    Output: (batch, embed_dim)
    """

    def __init__(self, n_channels: int, n_time: int, embed_dim: int = CNN_EMBEDDING_DIM):
        super().__init__()
        self.spatial_att = SpatialAttention(n_channels)
        self.temporal_att = TemporalAttention(n_time)
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, embed_dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_att(x)
        x = self.temporal_att(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class STAttModel(nn.Module):
    def __init__(self, n_channels: int, n_time: int, embed_dim: int = CNN_EMBEDDING_DIM, n_classes: int = 2):
        super().__init__()
        self.embedder = STAttEmbedder(n_channels, n_time, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x: torch.Tensor):
        emb = self.embedder(x)
        return self.classifier(emb), emb


def train_statt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = CNN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    embed_dim: int = CNN_EMBEDDING_DIM,
) -> STAttModel:
    if X_train.ndim == 2:
        X_train = X_train[:, np.newaxis, :]
    n_segments, n_channels, n_time = X_train.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STAttModel(n_channels, n_time, embed_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        logger.debug(f"STAtt Epoch {epoch+1}/{epochs} loss={total_loss/len(loader):.4f}")

    return model


def extract_statt_embeddings(model: STAttModel, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    if X.ndim == 2:
        X = X[:, np.newaxis, :]
    device = next(model.parameters()).device
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(device)
            _, emb = model(xb)
            embeddings.append(emb.cpu().numpy())
    return np.concatenate(embeddings, axis=0)
