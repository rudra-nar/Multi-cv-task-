import os
import numpy as np
from ultralytics import YOLO


model   = YOLO("yolov8n.pt")
results = model.predict(source=r"C:\Users\Asus\Documents\CV assignment 1\datasets\coco\images\val2017", save=False, verbose=False)

def compute_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

IOU_THRESHOLD = 0.1
confidences   = []
corrects      = []
#    results = model(source=..., stream=True)  # generator of Results objects
#    for r in results:
#        boxes = r.boxes  # Boxes object for bbox outputs
#        masks = r.masks  # Masks object for segment masks outputs
#        probs = r.probs  # Class probabilities for classification outputs

for result in results:
    fname      = os.path.splitext(os.path.basename(result.path))[0]   ## REFFERED LLM FOR FUNCTION
    label_file = os.path.join(r"C:\Users\Asus\Documents\CV assignment 1\datasets\coco\labels\val2017", fname + ".txt")

    if not os.path.exists(label_file):
        continue

    H, W = result.orig_shape   

    gt_boxes = []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:   
                continue

            class_id = int(parts[0])

            coords = list(map(float, parts[1:]))
            x_c, y_c, w, h = coords[0], coords[1], coords[2], coords[3]
            x1 = (x_c - w/2) * W
            y1 = (y_c - h/2) * H
            x2 = (x_c + w/2) * W
            y2 = (y_c + h/2) * H
            gt_boxes.append((class_id, [x1, y1, x2, y2]))

    if result.boxes is None or len(result.boxes) == 0:
        continue

    boxes  = result.boxes.xyxy.cpu().numpy()
    confidence = result.boxes.conf.cpu().numpy()
    cls    = result.boxes.cls.cpu().numpy().astype(int)

    matched_gts = set()
    for idx in np.argsort(-confidence):
        pred_box  = boxes[idx]
        pred_conf = confidence[idx]
        pred_cls  = cls[idx] 
        best_iou  = 0.0
        best_gt_i = -1
        for i, (gt_cls, gt_box) in enumerate(gt_boxes):
            if i in matched_gts or gt_cls != pred_cls:
                continue
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou  = iou
                best_gt_i = i

        is_correct = (best_iou >= IOU_THRESHOLD)
        if is_correct:
            matched_gts.add(best_gt_i)

        confidences.append(pred_conf)
        corrects.append(float(is_correct))

confidences = np.array(confidences)
corrects    = np.array(corrects)
n           = len(confidences)
NUM_BINS    = 10

ece  = 0.0
bins = np.linspace(0, 1, NUM_BINS + 1)

for i in range(NUM_BINS):
    lo, hi = bins[i], bins[i + 1]
    mask   = (confidences > lo) & (confidences <= hi)
    if mask.sum() == 0:
        continue
    ece += (mask.sum() / n) * abs(corrects[mask].mean() - confidences[mask].mean())

print(f"ECE = {ece:.4f}")