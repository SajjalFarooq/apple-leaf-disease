"""Train top-7 transfer-learning models, one by one, 5 epochs each (first pass).
No early stopping: all epochs run fully. Best-validation checkpoint saved
after every epoch, so a restart never loses completed epochs.
Models: resnet18, resnet50, efficientnet_b0, densenet121, mobilenet_v2, vgg16, efficientnet_b3
Seeds fixed: python 42, numpy 42, torch 42.
"""
import torch, torch.nn as nn, pathlib, json, time, random, sys
import numpy as np
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

random.seed(42); np.random.seed(42); torch.manual_seed(42)
BASE = pathlib.Path(__file__).parent
DATA = BASE / 'data'
OUT = BASE / 'model_results'; OUT.mkdir(exist_ok=True)
CLASSES = ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'Healthy']
EPOCHS = 5; BATCH = 32; LR = 0.001

tf = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
train_ds = datasets.ImageFolder(DATA / 'train', transform=tf)
val_ds = datasets.ImageFolder(DATA / 'validation', transform=tf)
test_ds = datasets.ImageFolder(DATA / 'test', transform=tf)
train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)
print(f"train {len(train_ds)} val {len(val_ds)} test {len(test_ds)}", flush=True)

def build(name):
    if name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, 4); return m, ['layer4', 'fc']
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1); m.fc = nn.Linear(m.fc.in_features, 4); return m, ['layer4', 'fc']
    if name == 'efficientnet_b0':
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4); return m, ['features.7', 'features.8', 'classifier']
    if name == 'efficientnet_b3':
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4); return m, ['features.7', 'features.8', 'classifier']
    if name == 'densenet121':
        m = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1); m.classifier = nn.Linear(m.classifier.in_features, 4); return m, ['features.denseblock4', 'classifier']
    if name == 'mobilenet_v2':
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 4); return m, ['features.18', 'classifier']
    if name == 'vgg16':
        m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1); m.classifier[6] = nn.Linear(m.classifier[6].in_features, 4); return m, ['features.28', 'features.29', 'features.30', 'classifier']
    raise ValueError(name)

def evaluate(model, dl):
    model.eval(); correct, total, preds, trues = 0, 0, [], []
    with torch.no_grad():
        for x, y in dl:
            p = model(x).argmax(1)
            correct += (p == y).sum().item(); total += len(y)
            preds.extend(p.tolist()); trues.extend(y.tolist())
    return correct/total, preds, trues

MODELS = sys.argv[1:] or ['resnet18', 'resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v2', 'vgg16', 'efficientnet_b3']
for name in MODELS:
    result_file = OUT / f"{name}_5epoch.json"
    if result_file.exists():
        print(f"SKIP {name}: already done", flush=True); continue
    print(f"\n=== MODEL {name} ===", flush=True)
    model, train_prefixes = build(name)
    for pname, p in model.named_parameters():
        if not any(pname.startswith(pref) for pref in train_prefixes):
            p.requires_grad = False
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    history, best_acc, best_state = [], 0, None
    for epoch in range(EPOCHS):
        model.train(); t0 = time.time(); running, seen = 0, 0
        for x, y in train_dl:
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step()
            running += loss.item()*len(y); seen += len(y)
        vacc, _, _ = evaluate(model, val_dl)
        history.append({'epoch': epoch+1, 'train_loss': running/seen, 'val_accuracy': vacc})
        print(f"{name} epoch {epoch+1}/{EPOCHS} loss={running/seen:.4f} val_acc={vacc:.4f} time={time.time()-t0:.0f}s", flush=True)
        if vacc >= best_acc:
            best_acc = vacc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, OUT / f"{name}_best.pth")
    model.load_state_dict(best_state)
    tacc, preds, trues = evaluate(model, test_dl)
    report = classification_report(trues, preds, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(trues, preds).tolist()
    result = {'model': name, 'epochs': EPOCHS, 'best_val_accuracy': best_acc,
              'test_accuracy': tacc, 'history': history,
              'classification_report': report, 'confusion_matrix': cm, 'classes': CLASSES}
    result_file.write_text(json.dumps(result, indent=2))
    print(f"{name} DONE test_acc={tacc:.4f} best_val={best_acc:.4f}", flush=True)
print("\nALL DONE", flush=True)
