import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import os
import json

# ---------------------------
# Transform
# ---------------------------
def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

# ---------------------------
# Model
# ---------------------------
def build_model(model_name='resnet50', num_classes=150, use_pretrained=True, fine_tune_all=True):
    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'efficientnet':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if use_pretrained else None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    else:
        raise ValueError("지원하지 않는 모델입니다.")

    # Fine-tuning 설정
    if not fine_tune_all:
        for param in model.parameters():
            param.requires_grad = False

        if model_name == 'resnet50':
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            for param in model.classifier[1].parameters():
                param.requires_grad = True

    return model

# ---------------------------
# Train & Evaluate
# ---------------------------
def train_and_evaluate(model, dataloaders, criterion, optimizer, num_epochs=5, device='cuda'):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc.item()

    return model, best_acc

# ---------------------------
# Main
# ---------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 상대경로 사용 (중요)
    data_path = "Archive/PokemonData"

    dataset = datasets.ImageFolder(data_path, transform=get_transforms())

    # 클래스 저장 (GUI용)
    classes = dataset.classes
    with open("classes.txt", "w") as f:
        for c in classes:
            f.write(c + "\n")

    # 데이터 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    num_classes = len(classes)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # 🔥 실험 4개
    # ---------------------------
    experiments = [
        {"name": "exp1", "model": "resnet50", "pretrained": True, "fine_tune_all": False},
        {"name": "exp2", "model": "resnet50", "pretrained": True, "fine_tune_all": True},
        {"name": "exp3", "model": "efficientnet", "pretrained": True, "fine_tune_all": True},
        {"name": "exp4", "model": "resnet50", "pretrained": False, "fine_tune_all": True},
    ]

    results = []

    for exp in experiments:
        print(f"\n🔥 Running {exp['name']}")

        model = build_model(
            model_name=exp["model"],
            num_classes=num_classes,
            use_pretrained=exp["pretrained"],
            fine_tune_all=exp["fine_tune_all"]
        ).to(device)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        model, best_acc = train_and_evaluate(
            model, dataloaders, criterion, optimizer, num_epochs=5, device=device
        )

        # 모델 저장
        torch.save(model.state_dict(), f"{exp['name']}.pth")

        results.append({
            "experiment": exp["name"],
            "model": exp["model"],
            "pretrained": exp["pretrained"],
            "fine_tune_all": exp["fine_tune_all"],
            "accuracy": best_acc
        })

    # 결과 저장 (README용)
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n📊 실험 결과:")
    for r in results:
        print(r)

    print("\n🎉 모든 실험 완료!")