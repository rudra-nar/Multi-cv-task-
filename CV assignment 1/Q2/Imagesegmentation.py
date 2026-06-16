import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torch.utils.data import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split  #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import os
import pandas as pd
from torchvision.io import decode_image
import wandb
from collections import Counter
torch.manual_seed(2024485)


import numpy as np
from torch import nn
from tqdm import tqdm
from torchvision.transforms import ToTensor
from torchvision.transforms import ToTensor


from sklearn.metrics import accuracy_score, f1_score, classification_report

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = [] 
        
        all_files = sorted(os.listdir(image_dir))
        
        for fname in all_files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(image_dir, fname)
            
            label_fname = fname.replace('.png', '_L.png')
            lbl_path = os.path.join(label_dir, label_fname)
            
            if os.path.isfile(img_path) and os.path.isfile(lbl_path):
                self.samples.append((img_path, lbl_path))
            else:
               
                print(f"Skipping {fname}: Label not found at {lbl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        

        image = image.resize((480, 360), Image.BILINEAR)
        label = label.resize((480, 360), Image.NEAREST) # MUST be NEAREST for masks

        if self.transform:
            image = self.transform(image)
            
        # Convert RGB mask to class index tensor (H, W)
        label_tensor = mask_to_label(label)
        
        return image, label_tensor

transform = transforms.Compose([transforms.Resize((480,360)),
                                transforms.ToTensor(),              # FIX 12: ToTensor was missing
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

datadset = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q2\data\train",
                              r"C:\Users\Asus\Documents\CV assignment 1\Q2\data\train_labels", 
                              transform=transform)

# ── CamVid class definitions ────────────────────────────────────────────────
# Standard CamVid 11-class colour → label mapping
CAMVID_CLASSES = [
    'Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 
    'CartLuggagePram', 'Child', 'Column_Pole', 'Fence', 'LaneMkgsDriv',
    'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter', 'OtherMoving',
    'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 
    'SignSymbol', 'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 
    'Train', 'Tree', 'Truck_Bus', 'Tunnel', 'VegetationMisc', 'Void', 'Wall'
]
CAMVID_COLORS = [(64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (128, 0, 0), 
                 (64, 0, 128), (64, 0, 192), (192, 128, 64), (192, 192, 128), (64, 64, 128), 
                 (128, 0, 192), (192, 0, 64), (128, 128, 64), (192, 0, 192), (128, 64, 64), (64, 192, 128),
                  (64, 64, 0), (128, 64, 128), (128, 128, 192), (0, 0, 192), (192, 128, 128), 
                  (128, 128, 128), (64, 128, 192), (0, 0, 64), (0, 64, 64), (192, 64, 128), 
                  (128, 128, 0), (192, 128, 192), (64, 0, 64), (192, 192, 0), (0, 0, 0), (64, 192, 0)]

NUM_CLASSES = len(CAMVID_CLASSES)

def mask_to_label(mask_pil):
    void_idx = CAMVID_CLASSES.index('Void') if 'Void' in CAMVID_CLASSES else 0
    mask_np = np.array(mask_pil)                    # (H, W, 3)  uint8
    label = torch.full(mask_np.shape[:2], void_idx, dtype=torch.long)
    for cls_idx, color in enumerate(CAMVID_COLORS):
        match = np.all(mask_np == color, axis=-1)   # (H, W) bool
        label[match] = cls_idx
    return label              # LongTensor (H, W)


# =============================================================3.1.a=========================================================================


transform = transforms.Compose([
    transforms.ToTensor(),                          
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_dataset = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q2\test\test_images",
                              r"C:\Users\Asus\Documents\CV assignment 1\Q2\test\test_labels", 
                              transform=transform)



train_dataloader = DataLoader(datadset, batch_size=4, shuffle=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=4, shuffle=False)

print(f"Train: {len(datadset)}   Test: {len(test_dataset)}")


# =============================================================3.1.b — class distribution=========================================================================
def get_pixel_counts(dataset):
    counts = Counter()
    for _, label in tqdm(dataset, desc="Counting pixels"):
        # label is a (H,W) LongTensor from __getitem__
        for cls_idx in range(NUM_CLASSES):
            counts[cls_idx] += (label == cls_idx).sum().item()
    return counts

def plot_class_distribution(counts, title):
    classes = CAMVID_CLASSES
    values  = [counts.get(i, 0) for i in range(NUM_CLASSES)]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(classes, values, color='steelblue')
    plt.xticks(rotation=30, ha='right')
    plt.ylabel('Pixel count')
    plt.title(title)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    fname = title.replace(' ', '_') + '.png'
    plt.savefig(fname, dpi=100)
    wandb.log({title: wandb.Image(fname)})
    plt.show()

wandb.init(project="CV-Assignment1-Segmentation", name="CamVid-DataExploration")

train_counts = get_pixel_counts(datadset)
test_counts  = get_pixel_counts(test_dataset)

plot_class_distribution(train_counts, 'Train Class Distribution')
plot_class_distribution(test_counts,  'Test Class Distribution')


# =============================================================3.1.c — 2 images + mask per class=========================================================================
def collect_samples_per_class(dataset, n=2):
    collected = {c: [] for c in range(NUM_CLASSES)}
    for img, label in dataset:
        for cls_idx in range(NUM_CLASSES):
            if len(collected[cls_idx]) < n and (label == cls_idx).any():
                collected[cls_idx].append((img, label))
        if all(len(v) >= n for v in collected.values()):
            break
    return collected


#LLM refenced for viszulization
def show_image_and_mask(img_tensor, label_tensor, cls_idx, ax_img, ax_mask):
    img_tensor = img_tensor.cpu()
    label_tensor = label_tensor.cpu()

    # Standard Unnormalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_unnorm = img_tensor * std + mean
    img_np = img_unnorm.permute(1, 2, 0).clamp(0, 1).numpy()

    h, w = label_tensor.shape
    mask_np = np.zeros((h, w, 3), dtype=np.uint8)
    for c, color in enumerate(CAMVID_COLORS):
        mask_np[label_tensor == c] = color

    ax_img.imshow(img_np)
    ax_mask.imshow(mask_np)

samples_per_class = collect_samples_per_class(datadset, n=2)
for cls_idx in range(NUM_CLASSES):
    pairs = samples_per_class[cls_idx]
    if not pairs:
        continue
 
    fig, axes = plt.subplots(len(pairs), 2, figsize=(8, 4 * len(pairs)))
    if len(pairs) == 1:
        axes = [axes]  # make iterable
 
    for row, (img, label) in enumerate(pairs):
        show_image_and_mask(img, label, cls_idx, axes[row][0], axes[row][1])
        axes[row][0].set_title(f"{CAMVID_CLASSES[cls_idx]} - Image")
        axes[row][1].set_title(f"{CAMVID_CLASSES[cls_idx]} - Mask")
 
    plt.tight_layout()
    fname = f'class_{cls_idx}_{CAMVID_CLASSES[cls_idx]}.png'
    plt.savefig(fname, dpi=100, bbox_inches='tight')
    wandb.log({f'samples_{CAMVID_CLASSES[cls_idx]}': wandb.Image(fname)})
    plt.close(fig)
 

print("All class samples have been logged to WandB!")
wandb.finish()