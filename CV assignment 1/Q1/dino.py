import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from torch.utils.data import Subset
import os
import wandb
import numpy as np

torch.manual_seed(2024485)

from transformers import AutoImageProcessor, AutoModel
from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

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
        path,label = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([transforms.Resize((224, 224)),
    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),])

vis_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

full = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q1\data",transform=transform)
loader = DataLoader(full, batch_size=32, shuffle=False)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2.eval()
dinov2 = dinov2.to(device)
labels = []
features = []

with torch.no_grad():
    for image, label in loader:
        image = image.to(device)
        feature = dinov2(image)
        features.append(feature.cpu().numpy())
        labels.extend(label.numpy()) 

all_features = np.concatenate(features, axis=0) 
all_labels   = np.array(labels)                  

class_names = list(class_mapping.keys())

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
features_2d = tsne.fit_transform(all_features)  




kmeans = KMeans(n_clusters=10, random_state=42)
cluster_labels = kmeans.fit_predict(all_features)

# REFFENCED LLM for plotting the data

plt.figure(figsize=(12, 8))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                      c=cluster_labels, cmap='tab10', alpha=0.6, s=15)
plt.colorbar(scatter, label='Cluster ID')
plt.title(" t-SNE projection")
plt.tight_layout()
plt.savefig("KMEAN.png", dpi=150)
plt.show()

# ── Visualize 16 images from cluster 0 ───────────────────────────────────────
vis_dataset = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q1\data", transform=vis_transform)

cluster_id   = 0
cluster_idxs = np.where(cluster_labels == cluster_id)[0][:16]

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
for ax, idx in zip(axes.flatten(), cluster_idxs):
    img, lbl = vis_dataset[idx]
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title(class_names[lbl], fontsize=8)
    ax.axis('off')
plt.suptitle(f"16 Images from Cluster {cluster_id}")
plt.tight_layout()
plt.savefig("cluster_images.png", dpi=150)
plt.show()