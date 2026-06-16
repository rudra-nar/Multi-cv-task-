import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from torch.utils.data import Subset
import os
import wandb
from sklearn.metrics import accuracy_score, f1_score, classification_report


torch.manual_seed(2024485)

from transformers import AutoImageProcessor, AutoModel
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd


# https://docs.wandb.ai/models/ref/python/custom-charts/confusion_matrix

import torchvision.models as models

from torch import nn
from tqdm import tqdm
from torchvision.transforms import ToTensor

#=============================================================2.1.a=========================================================================

##custom class 
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

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224))  # blank image for corrupt files
        if self.transform:
            image = self.transform(image)
        return image, label

#loading up dataset       


def train(dataloader, model, loss_fn, optimizer,epoch):
    model.train()
    """
    The above line is very important -- set the model to training mode.
    Must be used if you have previously invoked `model.eval()` to unfreeze the
    dropout and batchnorm layers that are disabled during evaluation.
    """
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
    wandb.log({
        "epoch": epoch,
        "train_loss": avg_loss,
        "train_accuracy": accuracy
    })
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
            all_pred.extend(predicted_classes.cpu().numpy())
            all_true.extend(y.cpu().numpy())  

    test_loss /= len(dataloader)
    accuracy = accuracy_score(all_true, all_pred)
    f1_macro = f1_score(all_true, all_pred, average='macro')
    f1_weighted = f1_score(all_true, all_pred, average='weighted')

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
    return test_loss, accuracy


#=============================================================2.2.b =========================================
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Slightly larger
    transforms.CenterCrop(224),      # Then crop to 224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]     # ImageNet std
    )
])
full = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q1\data",transform=transform)

# parameters to stratify
idx = list(range(len(full)))
labels = [sample[1] for sample in full.samples]   

#partitioning
trains,tests = train_test_split(idx, test_size = 0.2, stratify=labels)

train_dataset = Subset(full, trains)
val_dataset = Subset(full, tests)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

wandb.init(project="CV-Assignment1-Wildlife", name="Data-Setup")

train_labels = [full.samples[i][1] for i in trains]  
val_labels = [full.samples[i][1] for i in tests]   

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


model = models.resnet18(pretrained=True)
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(device)





learning_rate = 0.0003
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n{'-'*40}")
    train(train_dataloader, model, loss_fn, optimizer,epoch)
    test(test_dataloader, model, loss_fn,epoch)


#=============================================================2.2.b =========================================

#the model is not over fitting as the training loss and testing loss both go down as epoch increases

#=============================================================2.2.c =========================================

#done in test module itself 

#=============================================================2.2.d =========================================
