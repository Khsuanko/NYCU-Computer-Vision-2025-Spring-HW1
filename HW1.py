import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# Configurations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 20
learning_rate = 0.01
num_classes = 100

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Fix ImageFolder label sorting issue
def get_sorted_class_mapping(dataset):
    original_class_to_idx = dataset.class_to_idx
    
    sorted_classes = sorted(original_class_to_idx.keys(), key=lambda x: int(x))
    
    new_class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted_classes)}
    
    dataset.samples = [(path, new_class_to_idx[str(label)]) for path, label in dataset.samples]
    dataset.targets = [new_class_to_idx[str(label)] for _, label in dataset.samples]
    
    dataset.class_to_idx = new_class_to_idx
    
    return dataset

# Dataset and DataLoader
train_dataset = datasets.ImageFolder(root='data/train', transform=transform)
val_dataset = datasets.ImageFolder(root='data/val', transform=val_transform)

# Apply correct label mapping
train_dataset = get_sorted_class_mapping(train_dataset)
val_dataset = get_sorted_class_mapping(val_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Model setup
model = models.resnet50(pretrained=True)

# Freeze all layers initially
for param in model.parameters():
    param.requires_grad = False

# Unfreeze later layers for fine-tuning
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Modify final fully connected layer
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, num_classes)
)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.SGD([
    {'params': model.layer3.parameters(), 'lr': learning_rate/10},
    {'params': model.layer4.parameters(), 'lr': learning_rate/10},
    {'params': model.fc.parameters(), 'lr': learning_rate}
], momentum=0.9, weight_decay=1e-4, nesterov=True)

scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)

# Training loop
best_val_acc = 0.0
scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Mixed precision backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        progress_bar.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    epoch_time = time.time() - start_time
    train_accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}, "
          f"Accuracy: {train_accuracy:.2f}%, Time: {epoch_time:.2f}s")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    val_accuracy = 100 * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%")
    
    scheduler.step(val_accuracy)
    
    # Save best model
    if val_accuracy > best_val_acc:
        best_val_acc = val_accuracy
        torch.save(model.state_dict(), 'resnet50_best.pth')
        print(f"New best model saved with val_acc: {best_val_acc:.2f}%")


# Loading best model for testing
model.load_state_dict(torch.load('resnet50_best.pth'))

# Prediction on test set
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_images = os.listdir('data/test')
test_data = []

for img_name in tqdm(test_images, desc="Predicting test images"):
    img_path = os.path.join('data/test', img_name)
    image = datasets.folder.default_loader(img_path)
    image = test_transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        test_data.append((os.path.splitext(img_name)[0], predicted.item()))

# Create prediction.csv
submission_df = pd.DataFrame(test_data, columns=['image_name', 'pred_label'])
submission_df.to_csv('prediction.csv', index=False)

print("Prediction CSV generated: prediction.csv")
