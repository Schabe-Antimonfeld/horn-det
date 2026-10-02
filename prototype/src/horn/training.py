"""Training, validation calibration, safe checkpoints and model cost measurement."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np


def partitions(dataset):
    splits = np.asarray(dataset["splits"])
    groups = np.asarray(dataset["groups"])
    if not np.isin(splits, ["train", "validation", "test"]).all():
        raise ValueError("Freeze all dataset partitions before training")
    for group in np.unique(groups):
        if len(np.unique(splits[groups == group])) != 1:
            raise ValueError("Source group crossed partitions")
    result = {split: np.flatnonzero(splits == split) for split in ("train", "validation", "test")}
    if not result["train"].size or not result["validation"].size:
        raise ValueError("Training and validation samples are required")
    return result


class LinearSVM:
    def __init__(self):
        self.values = None

    def fit(self, dataset, seed=0):
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
        indices = partitions(dataset)
        features, labels = dataset["features"], dataset["labels"]
        if features.ndim != 2:
            raise ValueError("SVM requires MFCC statistics")
        train, validation = indices["train"], indices["validation"]
        if len(np.unique(labels[train])) != 2 or len(np.unique(labels[validation])) != 2:
            raise ValueError("SVM and probability calibration require both classes")
        scaler = StandardScaler().fit(features[train])
        model = LinearSVC(random_state=seed, max_iter=10000).fit(scaler.transform(features[train]), labels[train])
        scores = model.decision_function(scaler.transform(features[validation]))[:, None]
        calibration = LogisticRegression(random_state=seed).fit(scores, labels[validation])
        self.values = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": model.coef_[0],
                       "intercept": model.intercept_[0], "slope": calibration.coef_[0, 0],
                       "offset": calibration.intercept_[0], "frontend": np.asarray("mfcc")}
        if "config" in dataset:
            self.values["config"] = dataset["config"]
        return self

    def probability(self, features):
        from scipy.special import expit
        values = self.values
        if values is None:
            raise ValueError("SVM has not been fitted")
        score = ((np.asarray(features) - values["mean"]) / values["scale"]) @ values["coefficient"] + values["intercept"]
        return expit(values["slope"] * score + values["offset"])

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.values)

    @classmethod
    def load(cls, path):
        model = cls()
        with np.load(path, allow_pickle=False) as archive:
            model.values = dict(archive)
        return model


def train_model(model, dataset, epochs=10, batch_size=32, learning_rate=1e-3,
                seed=0, teacher_logits=None, temperature=2.0, mixture=0.5):
    import torch
    from .models import distillation_loss
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid training schedule")
    indices = partitions(dataset)
    features = torch.as_tensor(dataset["features"], dtype=torch.float32)
    labels = torch.as_tensor(dataset["labels"], dtype=torch.long)
    if teacher_logits is not None:
        teacher_logits = torch.as_tensor(teacher_logits, dtype=torch.float32)
        if teacher_logits.shape != (len(labels), 2):
            raise ValueError("Teacher logits must match student sample identities")
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)
    generator = np.random.default_rng(seed)
    history, best, chosen = [], float("inf"), None
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(indices["train"])
        losses = []
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(features[batch])
            loss = (torch.nn.functional.cross_entropy(logits, labels[batch]) if teacher_logits is None else
                    distillation_loss(logits, teacher_logits[batch], labels[batch], temperature, mixture))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.cross_entropy(model(features[indices["validation"]]), labels[indices["validation"]]))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": validation})
        if validation < best:
            best, chosen = validation, deepcopy(model.state_dict())
    model.load_state_dict(chosen)
    model.eval()
    return history


def save_model(model, path, name, config, frontend, history, metadata=None):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"state": model.state_dict(), "name": name, "config": asdict(config),
               "frontend": frontend, "history": history, "metadata": metadata or {}}
    torch.save(payload, path)


def model_cost(model, sample):
    """Count conv/linear MACs and parameters; GRU MACs are an analytical estimate."""
    import torch
    count, handles = [0], []

    def observe(module, inputs, output):
        if isinstance(module, torch.nn.Conv2d):
            count[0] += output.numel() * (module.in_channels // module.groups) * np.prod(module.kernel_size)
        elif isinstance(module, torch.nn.Linear):
            count[0] += output.numel() * module.in_features
        elif isinstance(module, torch.nn.GRU):
            values = inputs[0]
            count[0] += values.shape[0] * values.shape[1] * 3 * module.hidden_size * (module.input_size + module.hidden_size)

    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.GRU)):
            handles.append(module.register_forward_hook(observe))
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(sample)
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {"parameters": parameters, "float32_bytes": parameters * 4, "macs": int(count[0]),
            "mac_scope": "conv_linear_plus_estimated_gru; excludes frontend, normalization and activations"}
