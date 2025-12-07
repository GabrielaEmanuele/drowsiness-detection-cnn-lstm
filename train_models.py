import os
from pathlib import Path
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

MODE = "images" 

DATA_ROOT = Path("..") / "data"
SEQUENCES_DIR = DATA_ROOT / "sequences"
SEQUENCES_NPY = SEQUENCES_DIR / "sequences.npy"
LABELS_NPY = SEQUENCES_DIR / "labels.npy"
SEQUENCES_IMG_DIR = DATA_ROOT / "sequences_img"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 16 
EPOCHS = 30      
LR = 5e-4        
NUM_CLASSES = 2  # alerta / sonolento
EARLY_STOP_PATIENCE = 5 


def get_video_id_from_path(path: Path) -> int:
    name = path.stem
    parts = name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Nome inesperado de arquivo: {name}")
    video_str = parts[2] 
    return int(video_str.lstrip("0") or "0")


class FeatureSequenceDataset(Dataset):
    def __init__(self, seq_path: Path, labels_path: Path):
        if not seq_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError(f"Arquivos {seq_path} ou {labels_path} não encontrados.")
        self.sequences = np.load(seq_path) 
        self.labels = np.load(labels_path)  
        assert len(self.sequences) == len(self.labels), "Seqs e labels com tamanhos diferentes"

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = self.sequences[idx]
        y = self.labels[idx]    
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


class ImageSequenceDataset(Dataset):
    def __init__(self, root_dir: Path, allowed_video_ids=None, augment: bool = False):
        self.root_dir = root_dir
        self.augment = augment

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Pasta {self.root_dir} não encontrada.")

        all_files = sorted(self.root_dir.glob("*.pt"))
        if not all_files:
            raise RuntimeError(f"Nenhum .pt encontrado em {self.root_dir}")

        allowed_set = set(allowed_video_ids) if allowed_video_ids is not None else None
        self.files = []
        for f in all_files:
            vid = get_video_id_from_path(f)
            if (allowed_set is None) or (vid in allowed_set):
                self.files.append(f)

        if not self.files:
            if allowed_video_ids is not None:
                raise RuntimeError(
                    f"Nenhum .pt encontrado em {self.root_dir} "
                    f"para IDs {set(allowed_video_ids)}."
                )
            else:
                raise RuntimeError(f"Nenhum .pt utilizável encontrado em {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def _augment_video(self, video: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() < 0.5:
            video = torch.flip(video, dims=[3]) 

        if torch.rand(1).item() < 0.5:
            noise = torch.randn_like(video) * 0.02
            video = video + noise

        return video

    def __getitem__(self, idx):
        data = torch.load(self.files[idx])
        video = data["video"]  # (seq_len, 1, 64, 64)
        label = data["label"]  # 0 ou 1

        video = video.float()
        if self.augment:
            video = self._augment_video(video)

        label = torch.tensor(label, dtype=torch.long)
        return video, label


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # pega o último passo de tempo
        last_hidden = out[:, -1, :]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)
        return logits


class CNNLSTM(nn.Module):
    def __init__( self, feature_dim=128, hidden_size=64, num_layers=2, num_classes=2, dropout=0.5, bidirectional=True):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),             
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),        
        )

        self.fc = nn.Linear(128 * 8 * 8, feature_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size * num_directions, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, 1, 64, 64)
        batch, seq_len, C, H, W = x.size()
        # junta batch e tempo pra passar a CNN por frame
        x = x.view(batch * seq_len, C, H, W)
        feats = self.cnn(x)               
        feats = feats.view(batch * seq_len, -1)
        feats = self.fc(feats)       
        feats = self.dropout(feats)
        # volta pra (batch, seq_len, feature_dim)
        feats = feats.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(feats)
        last_hidden = lstm_out[:, -1, :] 
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        _, preds = outputs.max(1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = criterion(outputs, y)

            running_loss += loss.item() * X.size(0)
            _, preds = outputs.max(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    print(f"Modo selecionado: {MODE}")

    if MODE == "features":
        dataset = FeatureSequenceDataset(SEQUENCES_NPY, LABELS_NPY)

        sample_x, _ = dataset[0]
        seq_len, input_size = sample_x.shape
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout=0.5
        ).to(DEVICE)

        model_out_name = "best_lstm_features.pt"

        total = len(dataset)
        train_size = int(0.8 * total)
        val_size = total - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        test_dataset = None 

    elif MODE == "images":
        all_files = sorted(SEQUENCES_IMG_DIR.glob("*.pt"))
        if not all_files:
            raise RuntimeError(f"Nenhum .pt encontrado em {SEQUENCES_IMG_DIR}")

        all_ids = sorted({get_video_id_from_path(f) for f in all_files})
        random.shuffle(all_ids)
        print("Vídeos encontrados:", sorted(all_ids))
        n_videos = len(all_ids)

        n_train = max(1, int(0.7 * n_videos))
        n_val = max(1, int(0.15 * n_videos))
        n_test = n_videos - n_train - n_val
        if n_test <= 0:
            n_test = 1
            n_train = n_videos - n_val - n_test

        train_ids = all_ids[:n_train]
        val_ids = all_ids[n_train:n_train + n_val]
        test_ids = all_ids[n_train + n_val:]

        train_dataset = ImageSequenceDataset(SEQUENCES_IMG_DIR, allowed_video_ids=train_ids, augment=True)
        val_dataset = ImageSequenceDataset(SEQUENCES_IMG_DIR, allowed_video_ids=val_ids, augment=False)
        test_dataset = ImageSequenceDataset(SEQUENCES_IMG_DIR, allowed_video_ids=test_ids, augment=False)

        print(f"N seqs treino: {len(train_dataset)}, val: {len(val_dataset)}, teste: {len(test_dataset)}")

        sample_x, _ = train_dataset[0]
        seq_len, C, H, W = sample_x.shape
        print(f"[IMAGES] seq_len={seq_len}, C={C}, H={H}, W={W}")

        model = CNNLSTM(
            feature_dim=128,
            hidden_size=64,
            num_layers=2,
            num_classes=NUM_CLASSES,
            dropout=0.5,
            bidirectional=True
        ).to(DEVICE)

        model_out_name = "best_cnn_lstm_images.pt"

    else:
        raise ValueError("MODE inválido. Use 'features' ou 'images'.")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False) \
        if (MODE == "images" and test_dataset is not None) else None

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_val_acc = 0.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Época {epoch:02d}: "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
            print(f" >> novo melhor modelo (val_acc={best_val_acc:.3f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\nParando na época {epoch} (sem melhoria em {EARLY_STOP_PATIENCE} épocas).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        print(f"Test loss={test_loss:.4f} acc={test_acc:.3f}")

    model_dir = Path("..") / "model"
    model_dir.mkdir(exist_ok=True)

    out_path = model_dir / model_out_name
    if best_state is not None:
        torch.save(best_state, out_path)
        print(f"\nMelhor modelo salvo em: {out_path}")
    else:
        print("\nNenhum modelo melhor encontrado (??)")


if __name__ == "__main__":
    main()
