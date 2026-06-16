from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.val(data="coco.yaml", workers=0)

print(f"mAP50-95: {results.box.map:.4f}")
print(f"mAP50: {results.box.map50:.4f}")