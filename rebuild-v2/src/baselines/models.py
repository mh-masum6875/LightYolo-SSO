"""WP3 learned baselines (Master Plan 7 table). torch imports are function-local
(DICC env); anomaly models output a SCORE (higher = more anomalous), supervised
models output P(anomalous), so metrics.evaluate_scores applies uniformly.

  lstm_ae   2-layer LSTM AE, hidden 64, reconstruction-error score
  dense_ae  256-64-16-64-256 MLP AE on flattened windows
  cnn1d     3 conv blocks, kernel 5, supervised
  gru_clf   2-layer GRU, hidden 64, supervised
  ocsvm     RBF one-class SVM, nu tuned on validation, 12 window statistics
  iforest   isolation forest, 200 trees, same features
"""
from __future__ import annotations
import numpy as np

def _t():
    import torch, torch.nn as nn
    return torch, nn

def _seed_all(seed: int):
    import torch, random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _norm(wins):
    return (np.asarray(wins, dtype=np.float32) / 255.0)

class _TorchModelBase:
    epochs, batch, lr = 30, 256, 1e-3

    def _loader(self, X, y=None, shuffle=True):
        torch, _ = _t()
        tensors = [torch.tensor(X)] + ([torch.tensor(y)] if y is not None else [])
        ds = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(ds, batch_size=self.batch, shuffle=shuffle)

class LstmAE(_TorchModelBase):
    """Trains on NORMAL train windows only; score = reconstruction MSE."""
    supervised = False

    def fit(self, wins, labels, seed=0):
        torch, nn = _t(); _seed_all(seed)
        X = _norm(wins[labels == 0])
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.enc = nn.LSTM(4, 64, num_layers=2, batch_first=True)
                s.dec = nn.LSTM(64, 64, num_layers=2, batch_first=True)
                s.out = nn.Linear(64, 4)
            def forward(s, x):
                h, _ = s.enc(x)
                z = h[:, -1:, :].repeat(1, x.shape[1], 1)
                d, _ = s.dec(z)
                return s.out(d)
        self.net = Net()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(dev); self.dev = dev
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            for (xb,) in self._loader(X):
                xb = xb.to(dev); opt.zero_grad()
                loss = lossf(self.net(xb), xb); loss.backward(); opt.step()
        return self

    def score(self, wins):
        torch, _ = _t()
        X = _norm(wins); self.net.eval(); out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                xb = xb.to(self.dev)
                out.append(((self.net(xb) - xb) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out)

class DenseAE(LstmAE):
    """256-64-16-64-256 MLP AE on flattened windows (Master Plan 7)."""
    def fit(self, wins, labels, seed=0):
        torch, nn = _t(); _seed_all(seed)
        X = _norm(wins[labels == 0]).reshape(len(wins[labels == 0]), -1)
        d_in = X.shape[1]
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, d_in))
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(dev); self.dev = dev
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.MSELoss(); self.net.train()
        for _ in range(self.epochs):
            for (xb,) in self._loader(X):
                xb = xb.to(dev); opt.zero_grad()
                loss = lossf(self.net(xb), xb); loss.backward(); opt.step()
        return self

    def score(self, wins):
        torch, _ = _t()
        X = _norm(wins).reshape(len(wins), -1); self.net.eval(); out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                xb = xb.to(self.dev)
                out.append(((self.net(xb) - xb) ** 2).mean(dim=1).cpu().numpy())
        return np.concatenate(out)

class _SupervisedTorch(_TorchModelBase):
    supervised = True

    def _build(self):
        raise NotImplementedError

    def fit(self, wins, labels, seed=0):
        torch, nn = _t(); _seed_all(seed)
        X, y = _norm(wins), np.asarray(labels, dtype=np.int64)
        self.net = self._build()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(dev); self.dev = dev
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        lossf = nn.CrossEntropyLoss(); self.net.train()
        for _ in range(self.epochs):
            for xb, yb in self._loader(X, y):
                xb, yb = xb.to(dev), yb.to(dev); opt.zero_grad()
                loss = lossf(self.net(xb), yb); loss.backward(); opt.step()
        return self

    def score(self, wins):
        torch, _ = _t()
        X = _norm(wins); self.net.eval(); out = []
        with torch.no_grad():
            for (xb,) in self._loader(X, shuffle=False):
                xb = xb.to(self.dev)
                out.append(torch.softmax(self.net(xb), 1)[:, 1].cpu().numpy())
        return np.concatenate(out)

class Cnn1d(_SupervisedTorch):
    """3 conv blocks, kernel 5, supervised (input (B,64,4) -> (B,4,64))."""
    def _build(self):
        _, nn = _t()
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                def block(ci, co): return nn.Sequential(
                    nn.Conv1d(ci, co, 5, padding=2), nn.BatchNorm1d(co), nn.ReLU(), nn.MaxPool1d(2))
                s.f = nn.Sequential(block(4, 32), block(32, 64), block(64, 128),
                                    nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(128, 2))
            def forward(s, x): return s.f(x.permute(0, 2, 1))
        return Net()

class GruClf(_SupervisedTorch):
    def _build(self):
        _, nn = _t()
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.gru = nn.GRU(4, 64, num_layers=2, batch_first=True)
                s.fc = nn.Linear(64, 2)
            def forward(s, x):
                h, _ = s.gru(x)
                return s.fc(h[:, -1])
        return Net()

class OcSvm:
    """RBF one-class SVM on 12 window statistics; nu tuned on validation AUROC."""
    supervised = False
    NUS = (0.01, 0.05, 0.1, 0.2)

    def fit(self, wins, labels, seed=0, val=None):
        from sklearn.svm import OneClassSVM
        from src.baselines.features import batch_features
        from src.common.metrics import auroc
        Xtr = batch_features(wins[labels == 0])
        best = (None, -1.0)
        for nu in self.NUS:
            m = OneClassSVM(kernel="rbf", nu=nu, gamma="scale").fit(Xtr)
            if val is not None:
                vw, vy = val
                a = auroc(vy, -m.score_samples(batch_features(vw)))
                if a > best[1]:
                    best = (m, a); self.nu = nu
            else:
                best = (m, 0.0); self.nu = nu
        self.model = best[0]
        return self

    def score(self, wins):
        from src.baselines.features import batch_features
        return -self.model.score_samples(batch_features(wins))

class IForest:
    supervised = False

    def fit(self, wins, labels, seed=0, **_):
        from sklearn.ensemble import IsolationForest
        from src.baselines.features import batch_features
        self.model = IsolationForest(n_estimators=200, random_state=seed).fit(
            batch_features(wins[labels == 0]))
        return self

    def score(self, wins):
        from src.baselines.features import batch_features
        return -self.model.score_samples(batch_features(wins))

MODELS = {"lstm_ae": LstmAE, "dense_ae": DenseAE, "cnn1d": Cnn1d,
          "gru_clf": GruClf, "ocsvm": OcSvm, "iforest": IForest}
