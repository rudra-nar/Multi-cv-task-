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




#loading up dataset       
transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
full = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q1\data",transform=transform)

# parameters to stratify
idx = list(range(len(full)))
labels = [sample[1] for sample in full.samples]   

#partitioning
train,test = train_test_split(idx, test_size = 0.2, stratify=labels)

train_dataset = Subset(full, train)
val_dataset = Subset(full, test)


"""# Dictionary with hyperparameters
config = {
    'epochs' : 10,
    'lr' : 0.01
}

with wandb.init(project=project, config=config) as run:
    # Training code here
    # Log values to W&B with run.log()
    run.log({"accuracy": 0.9, "loss": 0.1})
"""
#=============================================================2.1.b =========================================================================

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

wandb.init(project="CV-Assignment1-Wildlife", name="Data-Setup")
#=============================================================2.1.c =========================================================================


train_labels = [full.samples[i][1] for i in train]  
val_labels = [full.samples[i][1] for i in test]   

# Count occurrences
train_counts = Counter(train_labels)
val_counts = Counter(val_labels)

print("Train distribution:", train_counts)
print("Val distribution:", val_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Get class names in order
class_names = list(class_mapping.keys())
class_indices = list(class_mapping.values())

# Train distribution
train_values = [train_counts[i] for i in class_indices]
ax1.bar(class_names, train_values)
ax1.set_title('Training Set Distribution')
ax1.set_xlabel('Class')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)  # Rotate labels

# Val distribution  
val_values = [val_counts[i] for i in class_indices]
ax2.bar(class_names, val_values)
ax2.set_title('Validation Set Distribution')
ax2.set_xlabel('Class')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

total_train = sum(train_values)
total_val = sum(val_values)

print("\nClass proportions:")
print("Class".ljust(15), "Train %".ljust(10), "Val %".ljust(10), "Difference")
print("-" * 50)

for i, name in enumerate(class_names):
    train_prop = (train_values[i] / total_train) * 100
    val_prop = (val_values[i] / total_val) * 100
    diff = abs(train_prop - val_prop)
    print(f"{name.ljust(15)} {train_prop:>6.2f}% {val_prop:>6.2f}% {diff:>6.2f}%")
    
#=============================================================2.2.a =========================================================================


train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.shape}")
print(f"Labels batch shape: {train_labels.shape}")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4,stride=4),

            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(14*14*128,10)
        )

    def forward(self, x):
        # This method is implicitly invoked when the model is called.
        x = self.features(x)
        logits = self.classification(x)
        return logits


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = NeuralNetwork().to(device)
print(model)

#=============================================================2.2.a =========================================

def train(dataloader, model, loss_fn, optimizer,epoch):
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
            all_true.extend(y.cpu().numpy())  # Store true labels

    # Calculations
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

learning_rate = 0.0009
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n{'-'*40}")
    train(train_dataloader, model, loss_fn, optimizer,epoch)
    test(test_dataloader, model, loss_fn,epoch)


#=============================================================2.2.c =========================================

# it is overfitting we can sat that because the eval loss is increasing after later epoch whereas the train loss reduces

#=============================================================2.2.d =========================================

## inbuilt in the tessting module itself


#=============================================================2.2.e =========================================

from collections import defaultdict


def misclassification(model, dataloader, device):
    model.eval()
    current_idx = 0
    misclassification = defaultdict(list)

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        predictions = output.argmax(dim = 1)

        for i in range(len(labels)):
            if(predictions[i] != labels[i]):
                misclassification[labels[i].item()].append((current_idx + i, labels[i].item(), predictions[i].item(), images[i].cpu()))

    samples = {}
    for class_id in range(10):  # 10 classes
        if class_id in misclassification and len(misclassification[class_id]) >= 3:
            samples[class_id] = misclassification[class_id][:3]  # First 3
        elif class_id in misclassification:
            samples[class_id] = misclassification[class_id]  # Less than 3 available
        else:
            print(f"Warning: No misclassifications found for class {class_id}")
    
    return samples
        
def visualization(samples,class_mapping):
    idx_to_class = {v: k for k, v in class_mapping.items()}
    
    # How many classes have misclassifications?
    num_classes = len(samples)
    
    # Create grid: rows = classes, cols = 3 images
    fig, axes = plt.subplots(num_classes, 3, figsize=(12, 4 * num_classes))
    
    # Handle edge case: only 1 class
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each misclassified image
    row_idx = 0
    for true_class, samples in sorted(samples.items()):
        for col_idx, (idx, true_label, pred_label, image) in enumerate(samples):
            
            # Convert tensor to numpy for display
            # Hint: tensor is [C, H, W], need [H, W, C]
            img_np = image.permute(1, 2, 0).numpy()
            
            # Clip to valid range (in case of numerical issues)
            img_np = np.clip(img_np, 0, 1)
            
            # Get the subplot
            if num_classes > 1:
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[col_idx]
            
            # Display image
            ax.imshow(img_np)
            
            # Set title with true and predicted class
            ax.set_title(
                f"True: {idx_to_class[true_label]}\n"
                f"Pred: {idx_to_class[pred_label]}",
                fontsize=9,
                color='red'
            )
            ax.axis('off')
        
        row_idx += 1
    
    plt.tight_layout()
    plt.savefig('misclassifications.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Log to wandb
    wandb.log({"misclassifications": wandb.Image(plt)})

samples = misclassification(model, test_dataloader, device)
visualization(samples,class_mapping)
    