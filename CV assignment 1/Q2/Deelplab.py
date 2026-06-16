import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models,transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb 
torch.manual_seed(2024485)

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

def mask_to_label(mask_pil):
    void_idx = CAMVID_CLASSES.index('Void') if 'Void' in CAMVID_CLASSES else 0
    mask_np = np.array(mask_pil)                    
    label = torch.full(mask_np.shape[:2], void_idx, dtype=torch.long)
    for cls_idx, color in enumerate(CAMVID_COLORS):
        match = np.all(mask_np == color, axis=-1)   
        label[match] = cls_idx
    return label             

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_path = self.samples[idx]
        image = Image.open(path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        
        image = image.resize((480, 360), Image.BILINEAR)
        label = label.resize((480, 360), Image.NEAREST)

        if self.transform:
            image = self.transform(image)
            
        label_tensor = mask_to_label(label)
        return image, label_tensor

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=32):
        super(DeepLabV3, self).__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
       
    def forward(self, x):
        return self.model(x)['out']

def train(model, data_loader, epoch, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    pixels = 0
    correct = 0
    
    bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    for step, (image, label) in enumerate(bar):  
        image = image.to(device)
        label = label.to(device)
        pred = model(image)
        loss = loss_fn(pred, label)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = pred.argmax(dim=1)

        correct += (prediction == label).sum().item()
        pixels += label.numel()
        bar.set_postfix({'loss': loss.item(), "acc": correct/pixels})

        wandb.log({
            "train/step_loss": loss.item(),
            "epoch": epoch,
            "step": epoch * len(data_loader) + step
        })
    
    avg_loss = total_loss / len(data_loader)
    avg_acc = correct / pixels

    wandb.log({
        "train/epoch_loss": avg_loss,
        "train/epoch_acc": avg_acc,
        "epoch": epoch
    })

    return avg_loss, avg_acc

def evaluation(model, loader, device):
    model.eval()
    
    accuracies = [[] for i in range(32)]
    ious = [[] for i in range(32)]
    dices = [[] for i in range(32)]
    precisions = [[] for i in range(32)]
    recalls = [[] for i in range(32)]
    
    total_correct = 0
    total_pixel = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="EVAL"):
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            predictions = predictions.argmax(dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_pixel += labels.numel()

            for prediction, label in zip(predictions, labels):
                pre = prediction.cpu().numpy()
                lab = label.cpu().numpy()

                for i in range(32):
                    predected_mask = (pre == i)
                    lab_mask = (lab == i)
                    
                    if not lab_mask.any():
                        continue

                    intersection = np.logical_and(lab_mask, predected_mask)
                    union = np.logical_or(lab_mask, predected_mask)
                    
                    iou = np.sum(intersection) / np.sum(union)
                    ious[i].append(iou)

                    dice = 2*np.sum(intersection)/(np.sum(predected_mask) + np.sum(lab_mask))
                    dices[i].append(dice)

                    pixel_acc = np.sum(predected_mask&lab_mask)/np.sum(lab_mask)
                    accuracies[i].append(pixel_acc)
                    
                    tp = np.sum(intersection)
                    fp = np.sum(predected_mask) - tp
                    fn = np.sum(lab_mask) - tp
                    
                    precision = tp/(tp+fp) if (tp+fp)>0 else 0
                    recall = tp/(tp+fn) if (tp+fn)>0 else 0
                    
                    precisions[i].append(precision)
                    recalls[i].append(recall)
    
    overall_acc = total_correct / total_pixel
    class_results = []
    valid_ious = []
    
    for i in range(32):
        if len(ious[i]) > 0:
            avg_iou = np.mean(ious[i])
            avg_dice = np.mean(dices[i])
            avg_precision = np.mean(precisions[i])
            avg_recall = np.mean(recalls[i])
            avg_pixel_acc = np.mean(accuracies[i])
            count = len(ious[i])
            valid_ious.append(avg_iou)
        else:
            avg_iou = 0
            avg_dice = 0
            avg_precision = 0
            avg_recall = 0
            avg_pixel_acc = 0
            count = 0
        
        class_results.append({
            'iou': avg_iou,
            'dice': avg_dice,
            'precision': avg_precision,
            'recall': avg_recall,
            'pixel_acc': avg_pixel_acc,
            'count': count
        })
    
    miou = np.mean(valid_ious) if valid_ious else 0

    print("\n========================================================================")
    print("EVALUATION RESULTS")
    print("========================================================================")
    print(f"\nOverall Pixel Accuracy: {overall_acc:.4f}")
    print(f"Mean IoU: {miou:.4f}")
    print(f"\n{'Class':<25} {'Pixel Acc':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'Samples':>8}")
    print("------------------------------------------------------------------------")
    
    for i, result in enumerate(class_results):
        print(f"{CAMVID_CLASSES[i]:<25} {result['pixel_acc']:>10.4f} {result['iou']:>8.4f} {result['dice']:>8.4f} "
              f"{result['precision']:>10.4f} {result['recall']:>8.4f} {result['count']:>8}")
    
    print("========================================================================")
    
    return overall_acc, miou, class_results

def find_low_iou(model, loader, device):
    model.eval()
    low_iou_samples = {}
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Finding low IoU"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            
            for i in range(images.size(0)):
                image = images[i].cpu()
                pred = predictions[i].cpu().numpy()
                label = labels[i].cpu().numpy()
                
                for class_id in range(32):
                    if class_id in low_iou_samples and len(low_iou_samples[class_id]) >= 3:
                        continue
                    
                    pred_mask = (pred == class_id)
                    label_mask = (label == class_id)
                    
                    if not label_mask.any():
                        continue
                    
                    intersection = np.logical_and(label_mask, pred_mask)
                    union = np.logical_or(label_mask, pred_mask)
                    iou = np.sum(intersection) / np.sum(union)
                    
                    if iou <= 0.5:
                        if class_id not in low_iou_samples:
                            low_iou_samples[class_id] = []
                        low_iou_samples[class_id].append({
                            'image': image,
                            'pred': pred,
                            'label': label,
                            'iou': iou
                        })
                
                classes_done = [c for c in low_iou_samples if len(low_iou_samples[c]) >= 3]
                if len(classes_done) >= 3:
                    break
            if len(classes_done) >= 3:
                break
    
    final = {}
    count = 0
    for class_id in sorted(low_iou_samples.keys()):
        if len(low_iou_samples[class_id]) >= 3:
            final[class_id] = low_iou_samples[class_id][:3]
            count += 1
            if count >= 3:
                break
    return final

def visualize_low_iou(low_iou_samples):
    if not low_iou_samples:
        print("No low IoU samples found")
        return
    
    num_classes = len(low_iou_samples)
    fig, axes = plt.subplots(num_classes, 9, figsize=(27, 3*num_classes))
    
    if num_classes == 1:
        axes = axes.reshape(1, -1)
    
    row = 0
    for class_id, samples in low_iou_samples.items():
        for col, sample in enumerate(samples):
            image = sample['image']
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1).permute(1, 2, 0).numpy()
            
            pred_colored = np.zeros((360, 480, 3), dtype=np.uint8)
            label_colored = np.zeros((360, 480, 3), dtype=np.uint8)
            
            for i, color in enumerate(CAMVID_COLORS):
                pred_colored[sample['pred'] == i] = color
                label_colored[sample['label'] == i] = color
            
            axes[row, col*3].imshow(image)
            axes[row, col*3].set_title(f"{CAMVID_CLASSES[class_id]}\nIoU={sample['iou']:.3f}", fontsize=9)
            axes[row, col*3].axis('off')
            
            axes[row, col*3+1].imshow(label_colored)
            axes[row, col*3+1].set_title('Ground Truth', fontsize=9)
            axes[row, col*3+1].axis('off')
            
            axes[row, col*3+2].imshow(pred_colored)
            axes[row, col*3+2].set_title('Prediction', fontsize=9)
            axes[row, col*3+2].axis('off')
        row += 1
    
    plt.tight_layout()
    plt.savefig('deeplabv3_low_iou.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to deeplabv3_low_iou.png")

def analyze_failures(low_iou_samples):
    print("\n========================================================================")
    print("FAILURE ANALYSIS")
    print("========================================================================")
    
    for class_id, samples in low_iou_samples.items():
        avg_iou = np.mean([s['iou'] for s in samples])
        print(f"\nClass: {CAMVID_CLASSES[class_id]}")
        print(f"  Average IoU: {avg_iou:.4f}")
        print(f"  Number of samples: {len(samples)}")
        
        if avg_iou < 0.2:
            print(f"  Failure Type: SEVERE MISCLASSIFICATION")
            print(f"  Possible Reasons:")
            print(f"    - Model confusing {CAMVID_CLASSES[class_id]} with visually similar classes")
            print(f"    - Insufficient training examples for this class")
            print(f"    - Similar color/texture to other objects in the scene")
            
        elif avg_iou < 0.4:
            print(f"  Failure Type: PARTIAL DETECTION / OCCLUSION")
            print(f"  Possible Reasons:")
            print(f"    - Object is partially occluded by other objects")
            print(f"    - Model detecting only visible portions")
            print(f"    - Boundary detection errors at object edges")
            print(f"    - Small object size making detection difficult")
            
        else:
            print(f"  Failure Type: BOUNDARY INACCURACIES")
            print(f"  Possible Reasons:")
            print(f"    - Model detecting object but with imprecise boundaries")
            print(f"    - Environmental confusion (similar surroundings)")
            print(f"    - Ambiguous object edges blending with background")
            print(f"    - Near-threshold predictions at class boundaries")

if __name__ == '__main__':
    wandb.init(
        project="deeplabv3-camvid",
        config={
            "epochs": 20,
            "batch_size": 4,
            "lr": 0.001,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "backbone": "resnet50",
            "pretrained": "COCO_WITH_VOC_LABELS_V1",
        }
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q2\data\train",
                                  r"C:\Users\Asus\Documents\CV assignment 1\Q2\data\train_labels", 
                                  transform=transform)
    Val_dataset = CustomImageDataset(r"C:\Users\Asus\Documents\CV assignment 1\Q2\test\test_images", 
                                      r"C:\Users\Asus\Documents\CV assignment 1\Q2\test\test_labels", 
                                      transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(Val_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = DeepLabV3()
    model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    best_miou = 0

    for epoch in range(epochs):
        loss, accuracy = train(model=model, data_loader=train_loader, optimizer=optimizer, 
                              device=device, epoch=epoch, loss_fn=loss_fn)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        if (epoch+1) % 5 == 0:
            val_acc, val_miou, _ = evaluation(model, val_loader, device)
            print(f"Val Acc={val_acc:.4f}, Val mIoU={val_miou:.4f}")

            wandb.log({"val/acc": val_acc, "val/miou": val_miou, "epoch": epoch})
            
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), 'best_deeplabv3.pth')
                print(f"Saved best model (mIoU={best_miou:.4f})")
    
    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load('best_deeplabv3.pth', weights_only=True))
    test_acc, test_miou, class_results = evaluation(model=model, loader=val_loader, device=device)

    wandb.log({"test/acc": test_acc, "test/miou": test_miou})
    
    print("\nFinding Low IoU Samples...")
    low_iou = find_low_iou(model, val_loader, device)
    
    if low_iou:
        print(f"\nFound low IoU samples for {len(low_iou)} classes")
        visualize_low_iou(low_iou)
        analyze_failures(low_iou)
    else:
        print("No low IoU samples found!")

    wandb.finish() 