import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import models,transforms
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


class SegNet_Encoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Encoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.MaxEn = nn.MaxPool2d(2, stride=2, return_indices=True) 

        self.ConvEn11 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.BNEn11 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvEn12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNEn12 = nn.BatchNorm2d(64, momentum=BN_momentum)

        self.ConvEn21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.BNEn21 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvEn22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNEn22 = nn.BatchNorm2d(128, momentum=BN_momentum)

        self.ConvEn31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.BNEn31 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvEn33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNEn33 = nn.BatchNorm2d(256, momentum=BN_momentum)

        self.ConvEn41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.BNEn41 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn43 = nn.BatchNorm2d(512, momentum=BN_momentum)

        self.ConvEn51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvEn53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNEn53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
    def forward(self,x):
        x = F.relu(self.BNEn11(self.ConvEn11(x))) 
        x = F.relu(self.BNEn12(self.ConvEn12(x))) 
        x, ind1 = self.MaxEn(x)
        size1 = x.size()

        x = F.relu(self.BNEn21(self.ConvEn21(x))) 
        x = F.relu(self.BNEn22(self.ConvEn22(x))) 
        x, ind2 = self.MaxEn(x)
        size2 = x.size()

        x = F.relu(self.BNEn31(self.ConvEn31(x))) 
        x = F.relu(self.BNEn32(self.ConvEn32(x))) 
        x = F.relu(self.BNEn33(self.ConvEn33(x)))   
        x, ind3 = self.MaxEn(x)
        size3 = x.size()

        x = F.relu(self.BNEn41(self.ConvEn41(x))) 
        x = F.relu(self.BNEn42(self.ConvEn42(x))) 
        x = F.relu(self.BNEn43(self.ConvEn43(x)))   
        x, ind4 = self.MaxEn(x)
        size4 = x.size()

        x = F.relu(self.BNEn51(self.ConvEn51(x))) 
        x = F.relu(self.BNEn52(self.ConvEn52(x))) 
        x = F.relu(self.BNEn53(self.ConvEn53(x)))   
        x, ind5 = self.MaxEn(x)
        size5 = x.size()
        return x,[ind1,ind2,ind3,ind4,ind5],[size1,size2,size3,size4,size5]
    
class SegNet_Decoder(nn.Module):
    def __init__(self, in_chn=3, out_chn=32, BN_momentum=0.5):
        super(SegNet_Decoder, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        
        self.MaxDe5 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.ConvDe53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe53 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe52 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe51 = nn.BatchNorm2d(512, momentum=BN_momentum)
        
        self.MaxDe4 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.ConvDe43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe43 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.BNDe42 = nn.BatchNorm2d(512, momentum=BN_momentum)
        self.ConvDe41 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.BNDe41 = nn.BatchNorm2d(256, momentum=BN_momentum)
        
        self.MaxDe3 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.ConvDe33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe33 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.BNDe32 = nn.BatchNorm2d(256, momentum=BN_momentum)
        self.ConvDe31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.BNDe31 = nn.BatchNorm2d(128, momentum=BN_momentum)
        
        self.MaxDe2 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.ConvDe22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.BNDe22 = nn.BatchNorm2d(128, momentum=BN_momentum)
        self.ConvDe21 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.BNDe21 = nn.BatchNorm2d(64, momentum=BN_momentum)
        
        self.MaxDe1 = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.ConvDe12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.BNDe12 = nn.BatchNorm2d(64, momentum=BN_momentum)
        self.ConvDe11 = nn.Conv2d(64, out_chn, kernel_size=3, padding=1) 

    def forward(self,x,indexes,sizes):
        ind1,ind2,ind3,ind4,ind5=indexes[0],indexes[1],indexes[2],indexes[3],indexes[4]
        size1,size2,size3,size4,size5=sizes[0],sizes[1],sizes[2],sizes[3],sizes[4]
        
        x = self.MaxDe5(x,ind5,output_size = size5)
        x = F.relu(self.BNDe53(self.ConvDe53(x)))
        x = F.relu(self.BNDe52(self.ConvDe52(x)))
        x = F.relu(self.BNDe51(self.ConvDe51(x)))
        
        x = self.MaxDe4(x,ind4,output_size = size4)
        x = F.relu(self.BNDe43(self.ConvDe43(x)))
        x = F.relu(self.BNDe42(self.ConvDe42(x)))
        x = F.relu(self.BNDe41(self.ConvDe41(x)))
        
        x = self.MaxDe3(x,ind3,output_size = size3)
        x = F.relu(self.BNDe33(self.ConvDe33(x)))
        x = F.relu(self.BNDe32(self.ConvDe32(x)))
        x = F.relu(self.BNDe31(self.ConvDe31(x)))
        
        x = self.MaxDe2(x,ind2,output_size = size2)
        x = F.relu(self.BNDe22(self.ConvDe22(x)))
        x = F.relu(self.BNDe21(self.ConvDe21(x)))
    
        x = self.MaxDe1(x,ind1,output_size = size1)
        x = F.relu(self.BNDe12(self.ConvDe12(x)))
        x = self.ConvDe11(x)
        return x

class SegNet_Pretrained(nn.Module):
    def __init__(self,encoder_weight_pth,in_chn=3, out_chn=32):
        super(SegNet_Pretrained, self).__init__()
        self.in_chn = in_chn
        self.out_chn = out_chn
        self.encoder=SegNet_Encoder(in_chn=self.in_chn,out_chn=self.out_chn)
        self.decoder=SegNet_Decoder(in_chn=self.in_chn,out_chn=self.out_chn)
        encoder_state_dict = torch.load(encoder_weight_pth,weights_only=True)
        self.encoder.load_state_dict(encoder_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self,x):
        input_size = x.size()
        x,indexes,sizes=self.encoder(x)
        sizes = [input_size, sizes[0], sizes[1], sizes[2], sizes[3]]

        x=self.decoder(x,indexes,sizes)
        return x

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

def train(model, train_loader, optmizer, device, epoch, loss_fn):
    model.train()
    total_loss = 0
    total_pixel = 0
    correct_pixel = 0
    
    bar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for step, (image, lable) in enumerate(bar): 
        image = image.to(device)
        label = lable.to(device)
        output = model(image)
        loss = loss_fn(output, label)
        
        total_loss += loss.item()
        
        optmizer.zero_grad()
        loss.backward()
        optmizer.step()
        
        prediction = output.argmax(dim=1)
        correct_pixel += (prediction == label).sum().item()
        total_pixel += label.numel()
        bar.set_postfix({'loss': loss.item(), "acc": correct_pixel/total_pixel})

        wandb.log({
            "train/step_loss": loss.item(),
            "epoch": epoch,
            "step": epoch * len(train_loader) + step
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = correct_pixel / total_pixel

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
        for images,labels in tqdm(loader, desc = "EVAL"):
            images,labels = images.to(device), labels.to(device)
            predictions = model(images)
            predictions = predictions.argmax(dim=1)
            
            total_correct += (predictions == labels).sum().item()
            total_pixel += labels.numel()

            for prediction, label in zip(predictions,labels):
                pre = prediction.cpu().numpy()
                lab = label.cpu().numpy()

                for i in range(32):
                    predected_mask = (pre == i)
                    lab_mask = (lab == i)
                    
                    if not lab_mask.any():
                        continue

                    intersection = np.logical_and(lab_mask,predected_mask)
                    union = np.logical_or(lab_mask,predected_mask)
                    
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
    plt.savefig('low_iou.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved to low_iou.png")

def analyze_failures(low_iou_samples):
    print("FAILURE ANALYSIS")
    
    for class_id, samples in low_iou_samples.items():
        avg_iou = np.mean([s['iou'] for s in samples])
        print(f"\n{CAMVID_CLASSES[class_id]}:")
        print(f"  Average IoU: {avg_iou:.4f}")
        
        if avg_iou < 0.2:
            print(f"  Reason: Severe misclassification - confused with similar classes")
        elif avg_iou < 0.4:
            print(f"  Reason: Partial detection - likely occlusion or boundary errors")
        else:
            print(f"  Reason: Boundary inaccuracies - environment causing confusion")

NUM_CLASSES = len(CAMVID_CLASSES)

if __name__ == '__main__':
    wandb.init(
        project="segnet-camvid",
        config={
            "epochs": 40,
            "batch_size": 4,
            "lr": 0.003,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "BN_momentum": 0.5,
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

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(Val_dataset, batch_size=4, shuffle=False, num_workers=2)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    encoder_path = r"C:\Users\Asus\Documents\CV assignment 1\Q2\encoder_model.pth"
    model = SegNet_Pretrained(encoder_path, in_chn=3, out_chn=32)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=.003)

    epochs = 20
    best_miou = 0

    for epoch in range(epochs):
        loss, accuracy = train(model=model, train_loader=train_loader, optmizer=optimizer, 
                              device=device, epoch=epoch, loss_fn=loss_fn)
        print(f"Epoch {epoch+1}: Loss={loss:.4f}, Acc={accuracy:.4f}")
        
        if (epoch+1) % 5 == 0:
            val_acc, val_miou, _ = evaluation(model, val_loader, device)
            print(f"Val Acc={val_acc:.4f}, Val mIoU={val_miou:.4f}")

            wandb.log({"val/acc": val_acc, "val/miou": val_miou, "epoch": epoch})
            
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"Saved best model (mIoU={best_miou:.4f})")

    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load('best_model.pth'))
    test_acc, test_miou, class_results = evaluation(model, val_loader, device)

    wandb.log({"test/acc": test_acc, "test/miou": test_miou})

    print("\n ========================================================================")
    print("EVALUATION RESULTS")
    print("=========================================================================")
    print(f"\nOverall Pixel Accuracy: {test_acc:.4f}")
    print(f"Mean IoU: {test_miou:.4f}")
    print(f"\n{'Class':<25} {'Pixel Acc':>10} {'IoU':>8} {'Dice':>8} {'Precision':>10} {'Recall':>8} {'Samples':>8}")
    print("=======================================================================")
    
    for i, result in enumerate(class_results):
        print(f"{CAMVID_CLASSES[i]:<25} {result['pixel_acc']:>10.4f} {result['iou']:>8.4f} {result['dice']:>8.4f} "
              f"{result['precision']:>10.4f} {result['recall']:>8.4f} {result['count']:>8}")
    print("="*110)

    low_iou = find_low_iou(model, val_loader, device)
    if low_iou:
        visualize_low_iou(low_iou)
        analyze_failures(low_iou)

    wandb.finish()