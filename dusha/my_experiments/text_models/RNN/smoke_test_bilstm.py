"""Минимальный smoke-тест для проверки forward-pass модели BiLSTM_Text."""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent))
from BiLSTM_Text import TextBiLSTMClassifier


def run_smoke_test() -> None:
    model = TextBiLSTMClassifier(
        vocab_size=128,
        embed_dim=32,
        hidden_size=16,
        lstm_layers=1,
        lstm_dropout=0.0,
        classifier_dropout=0.1,
        bidirectional=True,
        n_classes=4,
    )
    model.eval()

    input_ids = torch.tensor(
        [
            [2, 5, 6, 7, 0, 0],
            [4, 9, 11, 12, 13, 14],
            [8, 0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    lengths = torch.tensor([4, 6, 1], dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, lengths)

    assert logits.shape == (3, 4), f"Ожидалась форма (3, 4), получено {tuple(logits.shape)}"
    print("Smoke test passed: logits shape =", tuple(logits.shape))


if __name__ == "__main__":
    run_smoke_test()
