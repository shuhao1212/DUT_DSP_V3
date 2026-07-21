"""
train_13class.py — 迁移训练 13 类模型 (加唤醒词 marvin)

从 12 类旧模型迁移: 复用 backbone (前 8 层), FC 层随机初始化.
"""

import sys, torch, numpy as np, time, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import BcResNet, count_parameters
from load_weights import parse_weights_c, load_weights_to_model
from data_loader import create_precomputed_dataloaders, LABELS, COMMAND_WORDS
from export_weights import export_weights

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_C = WORKSPACE_ROOT / "Code" / "User" / "weights.c"
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_13class"


def migrate_weights(old_model_12, new_model_13):
    """将 12 类模型的 backbone 权重复制到 13 类模型, FC 层保持随机."""
    old_state = old_model_12.state_dict()
    new_state = new_model_13.state_dict()

    for name, param in new_state.items():
        if name in old_state and old_state[name].shape == param.shape:
            new_state[name] = old_state[name].clone()
        elif 'fc' in name:
            print(f"  [SKIP] {name}: keeping random init (new class)")
        else:
            print(f"  [WARN] {name}: shape mismatch or missing in old model")

    new_model_13.load_state_dict(new_state)
    return new_model_13


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch:3d} | Batch {batch_idx:4d}/{len(loader):4d} | "
                  f"Loss: {loss.item():.4f} | Acc: {100.0 * correct / total:.1f}%")
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        all_preds.extend(predicted.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())
    return running_loss / len(loader), 100.0 * correct / total, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 1: 加载旧 12 类权重
    print("Loading old 12-class weights ...")
    old_model = BcResNet(n_classes=12)
    weights_dict = parse_weights_c(WEIGHTS_C)
    old_model = load_weights_to_model(old_model, weights_dict)
    old_model.eval()

    # Step 2: 创建 13 类模型, 迁移 backbone
    print("Creating 13-class model and migrating backbone ...")
    model = BcResNet(n_classes=13).to(device)
    model = migrate_weights(old_model, model)
    print(f"Model params: {count_parameters(model)}")

    # Step 3: 数据
    print("Loading data ...")
    train_loader, val_loader, test_loader = create_precomputed_dataloaders(
        batch_size=args.batch_size, augment=True, num_workers=4)

    # Step 4: 优化器 (FC 层用更高学习率)
    fc_params = [p for n, p in model.named_parameters() if 'fc' in n]
    other_params = [p for n, p in model.named_parameters() if 'fc' not in n]
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr * 0.1},
        {'params': fc_params, 'lr': args.lr * 3.0},  # FC lr x3
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss()

    # Step 5: 训练
    best_val_acc = 0.0
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Training 13-class model: {args.epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_preds, val_targets = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch:3d} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_acc': val_acc},
                       CHECKPOINT_DIR / "best_model.pt")
            print(f"  >>> Saved best (val={val_acc:.2f}%)")

        if epoch % 5 == 0:
            per_class = {}
            preds_arr = np.array(val_preds)
            targets_arr = np.array(val_targets)
            for c in range(13):
                mask = targets_arr == c
                if mask.sum() > 0:
                    per_class[LABELS[c]] = 100.0 * (preds_arr[mask] == c).sum() / mask.sum()
            print("  Per-class (val):")
            for lbl, acc in per_class.items():
                m = "⚠️ " if acc < 60 else "  "
                print(f"    {m}{lbl:>12}: {acc:5.1f}%")

    # Step 6: 测试
    print(f"\n{'='*60}")
    print(f"Best val_acc: {best_val_acc:.2f}%")
    ckpt = torch.load(CHECKPOINT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    test_loss, test_acc, test_preds, test_targets = evaluate(model, test_loader, criterion, device)
    print(f"Test Acc: {test_acc:.2f}%")

    # 每类准确率
    print("\nPer-class (test):")
    preds_arr = np.array(test_preds)
    targets_arr = np.array(test_targets)
    for c in range(13):
        mask = targets_arr == c
        if mask.sum() > 0:
            acc = 100.0 * (preds_arr[mask] == c).sum() / mask.sum()
            m = "❌" if acc < 50 else ("⚠️" if acc < 80 else "✅")
            print(f"  {m} {LABELS[c]:>12}: {acc:5.1f}% ({mask.sum()} samples)")

    # 导出权重
    final_c = CHECKPOINT_DIR / "weights_13class.c"
    final_h = CHECKPOINT_DIR / "weights_13class.h"
    export_weights(model, output_c=final_c, output_h=final_h, backup=False)
    print(f"\nWeights exported to {final_c}")


if __name__ == "__main__":
    main()
