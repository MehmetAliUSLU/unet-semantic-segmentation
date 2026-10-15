import torch

from segmentation import DiceCELoss, DiceLoss, SegmentationMetrics, UNet
from segmentation.data import SyntheticShapes


def test_unet_output_shape():
    model = UNet(3, 5, base_channels=8)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5, 64, 64)


def test_unet_handles_odd_input_size():
    model = UNet(3, 2, base_channels=8)
    assert model(torch.randn(1, 3, 75, 93)).shape == (1, 2, 75, 93)


def test_dice_loss_is_near_zero_for_perfect_prediction():
    target = torch.randint(0, 3, (2, 16, 16))
    logits = torch.nn.functional.one_hot(target, 3).permute(0, 3, 1, 2).float() * 100
    assert DiceLoss()(logits, target).item() < 1e-3
    assert DiceCELoss()(logits, target).item() < 1e-3


def test_metrics_perfect_and_known_case():
    m = SegmentationMetrics(2)
    target = torch.tensor([[[0, 0], [1, 1]]])
    logits = torch.nn.functional.one_hot(target, 2).permute(0, 3, 1, 2).float()
    m.update(logits, target)
    assert m.compute()["mean_iou"] == 1.0

    m.reset()
    pred_logits = torch.nn.functional.one_hot(torch.tensor([[[0, 1], [1, 1]]]), 2)
    m.update(pred_logits.permute(0, 3, 1, 2).float(), target)
    r = m.compute()
    # class 0: tp=1, fn=1 -> IoU 0.5 ; class 1: tp=2, fp=1 -> IoU 2/3
    assert abs(r["per_class_iou"][0] - 0.5) < 1e-6
    assert abs(r["per_class_iou"][1] - 2 / 3) < 1e-6
    assert r["pixel_acc"] == 0.75


def test_synthetic_dataset_is_deterministic():
    ds = SyntheticShapes(length=4, size=32)
    (a, ma), (b, mb) = ds[1], ds[1]
    assert torch.equal(a, b) and torch.equal(ma, mb)
    assert a.shape == (3, 32, 32) and ma.shape == (32, 32)
    assert ma.max() <= 2


def test_model_overfits_a_tiny_batch():
    torch.manual_seed(0)
    ds = SyntheticShapes(length=4, size=32)
    x = torch.stack([ds[i][0] for i in range(4)])
    y = torch.stack([ds[i][1] for i in range(4)])
    model = UNet(3, 3, base_channels=8, depth=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = DiceCELoss()
    first = loss_fn(model(x), y).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
