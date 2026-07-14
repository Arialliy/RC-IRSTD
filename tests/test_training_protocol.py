from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")


def test_deterministic_global_max_pool_matches_adaptive_pool_tie_rule() -> None:
    from model.MSHNet import DeterministicGlobalMaxPool2d

    adaptive_input = torch.tensor(
        [[[[1.0, 2.0, 2.0], [0.0, 2.0, 1.0]]]],
        requires_grad=True,
    )
    deterministic_input = adaptive_input.detach().clone().requires_grad_(True)
    expected = torch.nn.functional.adaptive_max_pool2d(adaptive_input, 1)
    actual = DeterministicGlobalMaxPool2d()(deterministic_input)

    assert torch.equal(actual, expected)
    expected.sum().backward()
    actual.sum().backward()
    assert torch.equal(deterministic_input.grad, adaptive_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mshnet_strict_deterministic_cuda_backward() -> None:
    from model.MSHNet import MSHNet

    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        model = MSHNet(3).to("cuda").train()
        inputs = torch.randn(2, 3, 32, 32, device="cuda")
        auxiliary, output = model(inputs, True)
        loss = output.square().mean() + sum(
            prediction.square().mean() for prediction in auxiliary
        )
        loss.backward()
        assert all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled,
            warn_only=previous_warn_only,
        )


def test_validation_role_never_falls_back_to_official_test(tmp_path) -> None:
    from utils.data import IRSTD_Dataset

    (tmp_path / "test.txt").write_text("one.png\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="mode.*val"):
        IRSTD_Dataset._find_split_file(str(tmp_path), "val")
    assert IRSTD_Dataset._find_split_file(str(tmp_path), "test").endswith("test.txt")


def test_legacy_trainer_default_does_not_construct_test_loader(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main as legacy_main

    requested_modes: list[str] = []

    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, _args, mode: str) -> None:
            requested_modes.append(mode)
            self.list_dir = tmp_path / f"{mode}.txt"

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int):
            return torch.zeros(3, 16, 16), torch.zeros(1, 16, 16)

    class DummyModel(torch.nn.Module):
        def __init__(self, _channels: int) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(()))

        def forward(self, value, _tag):
            prediction = value[:, :1] * self.weight
            return [], prediction

    monkeypatch.setattr(legacy_main, "IRSTD_Dataset", DummyDataset)
    monkeypatch.setattr(legacy_main, "MSHNet", DummyModel)
    args = SimpleNamespace(
        mode="train",
        device="cpu",
        base_size=16,
        crop_size=16,
        batch_size=1,
        num_workers=0,
        multi_gpus=False,
        lr=0.05,
        warm_epoch=5,
        if_checkpoint=False,
        save_dir=str(tmp_path / "runs"),
        allow_test_selection=False,
        seed=7,
    )
    trainer = legacy_main.Trainer(args)
    assert requested_modes == ["train"]
    assert trainer.val_loader is None
