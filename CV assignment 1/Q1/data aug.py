import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import os
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import defaultdict

torch.manual_seed(2024485)

from transformers import AutoImageProcessor, AutoModel
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd

import torchvision.models as models

from torch import nn
from tqdm import tqdm
from torchvision.transforms import ToTensor

from torch.utils.data import ConcatDataset

#=============================================================2.1.a=========================================================================

class_mapping = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear':
4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.samples = []
        self.class_mapping = {'amur_leopard': 0, 'amur_tiger': 1, 'birds': 2, 'black_bear': 3, 'brown_bear':
4, 'dog': 5, 'roe_deer': 6, 'sika_deer': 7, 'wild_boar': 8, 'people': 9}
        
        for class_name in os.listdir(img_dir):
            path = os.path.join(img_dir,class_name)
            if not os.path.isdir(path):
                continue
            label = self.class_mapping[class_name]
            for image in os.listdir(path):
                self.samples.append((os.path.join(path,image), label))
        
    def __len__(self):
        return len(self.samples)
    
    def item(self,i):
        return self.samples[i]

    def __getitem__(self, idx):
        path,label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


# =============================================================2.4.a=========================================================================


aug1 = transforms.Compose([                        # Horizontal Flip
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=1.0),      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), 
])

aug2 = transforms.Compose([                        # Color Jitter
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3,
                           saturation=0.3, hue=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]), 
])

aug3 = transforms.Compose([                        # Grayscale
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def augumentation_sample(dataset, number):

    samples = []
    classes = {}
    for i in range(len(dataset)):
        path, label = dataset.item(i)             
        if classes.get(label, 0) >= number:
            continue
        image = Image.open(path).convert('RGB')
        samples.append((aug1(image), label))
        samples.append((aug2(image), label))
        samples.append((aug3(image), label))
        classes[label] = classes.get(label, 0) + 1  
    return samples


class AugmentedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


## LLM used in vizualizing the missclassified images
def visualize_augmentations(dataset, n_images=5):
    indices = list(range(0, len(dataset), max(1, len(dataset) // n_images)))[:n_images]
    fig, axes = plt.subplots(n_images, 4, figsize=(14, 3 * n_images))
    aug_labels = ['Original', 'HFlip', 'ColorJitter', 'Grayscale']

    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    base_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    for row, idx in enumerate(indices):
        path, label = dataset.item(idx)
        pil_img = Image.open(path).convert('RGB')
        versions = [base_tf(pil_img), aug1(pil_img), aug2(pil_img), aug3(pil_img)]
        for col, (tensor, title) in enumerate(zip(versions, aug_labels)):
            img = inv_normalize(tensor).permute(1, 2, 0).clamp(0, 1).numpy()
            axes[row][col].imshow(img)
            axes[row][col].set_title(f'{title}\n(class {label})', fontsize=8)
            axes[row][col].axis('off')

    plt.tight_layout()
    plt.savefig('augmentation_samples.png', dpi=100, bbox_inches='tight')
    wandb.log({"augmentation_samples": wandb.Image('augmentation_samples.png')})
    plt.show()


# =====================================================================================

def train(dataloader, model, loss_fn, optimizer, epoch):
    model.train()
    size = len(dataloader.dataset)
    total_loss = 0
    correct = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / size
    wandb.log({"epoch": epoch, "train_loss": avg_loss, "train_accuracy": accuracy})
    return avg_loss, accuracy


def test(dataloader, model, loss_fn, epoch):
    model.eval()
    test_loss, correct = 0, 0
    all_pred = []
    all_true = []

    with torch.no_grad():
        for X, y in tqdm(dataloader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            predicted_classes = pred.argmax(1)
            all_pred.extend(predicted_classes.cpu().numpy())  # FIX: was .device().numpy()
            all_true.extend(y.cpu().numpy())

    test_loss /= len(dataloader)
    accuracy = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average='macro')
    f1_weighted = f1_score(all_true, all_pred, average='weighted')

    # 2.4.d  Confusion matrix + metrics
    wandb.log({
        "epoch": epoch,
        "val_loss": test_loss,
        "val_accuracy": accuracy,
        "val_f1_macro": f1_macro,
        "val_f1_weighted": f1_weighted,
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_true,
            preds=all_pred,
            class_names=list(class_mapping.keys())
        )
    })

    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, F1 (Macro): {f1_macro:>8f}, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy, all_true, all_pred



#LLM reffrenced 
def get_misclassifications(dataloader, model):
    model.eval()
    current_idx = 0
    misclassification = defaultdict(list)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        predictions = output.argmax(dim=1)

        for i in range(len(labels)):
            if predictions[i] != labels[i]:
                misclassification[labels[i].item()].append(
                    (current_idx + i, labels[i].item(), predictions[i].item(), images[i].cpu())
                )
        current_idx += len(labels)

    samples = {}
    for class_id in range(10):
        if class_id in misclassification and len(misclassification[class_id]) >= 3:
            samples[class_id] = misclassification[class_id][:3]
        elif class_id in misclassification:
            samples[class_id] = misclassification[class_id]
        else:
            print(f"Warning: No misclassifications found for class {class_id}")
    return samples


#=============================================================2.2.b=========================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # FIX: std was missing
])

full = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q1\data", transform=transform)

idx = list(range(len(full)))
labels = [sample[1] for sample in full.samples]

trains, tests = train_test_split(idx, test_size=0.2, stratify=labels)

train_dataset = Subset(full, trains)
val_dataset = Subset(full, tests)

#=============================================================2.4.a=========================================================================

aug_samples = augumentation_sample(full, number=50)
aug_dataset = AugmentedDataset(aug_samples)
train_dataset_aug = ConcatDataset([train_dataset, aug_dataset])

print(f"Original train size : {len(train_dataset)}")

train_dataloader = DataLoader(train_dataset_aug, batch_size=32, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

wandb.init(project="CV-Assignment1-Wildlife", name="ResNet18-Augmented")


visualize_augmentations(full, n_images=5)

#=============================================================2.4.b=========================================================================
model = models.resnet18(pretrained=True)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

learning_rate = 0.0003
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
train_losses, val_losses = [], []

for epoch in range(epochs):
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer, epoch)
    val_loss, val_acc, all_true, all_pred = test(test_dataloader, model, loss_fn, epoch)
    train_losses.append(train_loss)
    val_losses.append(val_loss)




#=============================================================2.4.d — Accuracy, F1, confusion matrix=========================================================================

print(classification_report(all_true, all_pred, target_names=list(class_mapping.keys())))

wandb.finish() 