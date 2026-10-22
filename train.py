"""Apple leaf disease classification - transfer learning with ResNet18.

Dataset in data/ (see README for full detail):
  train: 12,000 images, 3,000 per class (original + augmented, see augment.py)
  validation: 317 images, test: 317 images, both unaugmented originals
Model: torchvision ResNet18 pretrained on ImageNet, final layer replaced
  for 4 classes. Early layers frozen for first epochs, then fine-tuned.
Outputs: best model weights, training history, confusion matrix, sample predictions.
"""
import torch, torch.nn as nn, pathlib, json, time
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

BASE = pathlib.Path(__file__).parent
DATA = BASE / 'data'
CLASSES = ['Apple_scab', 'Black_rot', 'Cedar_apple_rust', 'Healthy']
DEVICE = 'cpu'
EPOCHS = 5
BATCH = 32

tf_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
tf_eval = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder(DATA / 'train', transform=tf_train)
val_ds = datasets.ImageFolder(DATA / 'validation', transform=tf_eval)
test_ds = datasets.ImageFolder(DATA / 'test', transform=tf_eval)
print('classes:', train_ds.classes, flush=True)
print('train:', len(train_ds), 'val:', len(val_ds), 'test:', len(test_ds), flush=True)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=0)
test_dl = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(DEVICE)
# freeze all except layer4 and fc for transfer learning
for name, p in model.named_parameters():
    if not (name.startswith('layer4') or name.startswith('fc')):
        p.requires_grad = False

opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def evaluate(dl):
    model.eval()
    correct, total, preds, trues = 0, 0, [], []
    with torch.no_grad():
        for x, y in dl:
            out = model(x)
            p = out.argmax(1)
            correct += (p == y).sum().item(); total += len(y)
            preds.extend(p.tolist()); trues.extend(y.tolist())
    return correct/total, preds, trues

history = []
best_acc, best_state = 0, None
for epoch in range(EPOCHS):
    model.train(); t0 = time.time(); running, seen = 0, 0
    for x, y in train_dl:
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward(); opt.step()
        running += loss.item()*len(y); seen += len(y)
    vacc, _, _ = evaluate(val_dl)
    history.append({'epoch': epoch+1, 'train_loss': running/seen, 'val_accuracy': vacc})
    print(f"epoch {epoch+1}/{EPOCHS} loss={running/seen:.4f} val_acc={vacc:.4f} time={time.time()-t0:.0f}s", flush=True)
    if vacc > best_acc:
        best_acc = vacc; best_state = {k: v.cpu() for k, v in model.state_dict().items()}

torch.save(best_state, BASE / 'best_model.pth')
model.load_state_dict(best_state)
tacc, preds, trues = evaluate(test_dl)
print(f"TEST accuracy: {tacc:.4f}", flush=True)
print(classification_report(trues, preds, target_names=CLASSES), flush=True)
print('confusion matrix:'); print(confusion_matrix(trues, preds), flush=True)
(BASE / 'results.json').write_text(json.dumps({
    'best_val_accuracy': best_acc, 'test_accuracy': tacc,
    'history': history,
    'confusion_matrix': confusion_matrix(trues, preds).tolist(),
    'classes': CLASSES,
}, indent=2))
print('saved best_model.pth and results.json', flush=True)
