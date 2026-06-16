import os, json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from tidecv import TIDE, Data

IMG_DIR   = r"C:\Users\Asus\Documents\CV assignment 1\datasets\coco\images\val2017"
LABEL_DIR = r"C:\Users\Asus\Documents\CV assignment 1\datasets\coco\labels\val2017"

def get_scale(area):
    if area < 36*36:   return "small"
    if area < 96*96:   return "medium"
    return "large"

def iou(a, b):
    ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter) if (ua+ub-inter)>0 else 0

model   = YOLO("yolov8n.pt")
results = model.predict(source=IMG_DIR, save=False, verbose=False, stream=True, device='cpu')

preds, gts = [], []
for result in results:
    fname    = os.path.splitext(os.path.basename(result.path))[0]
    image_id = int(fname) if fname.isdigit() else abs(hash(fname))
    H, W     = result.orig_shape

    lf = os.path.join(LABEL_DIR, fname + ".txt")
    if os.path.exists(lf):
        for line in open(lf):
            p = line.split()
            if len(p) < 5: continue
            c = list(map(float, p[1:]))
            xs=[c[i]*W for i in range(0,len(c),2)]; ys=[c[i]*H for i in range(1,len(c),2)]
            box = [min(xs),min(ys),max(xs),max(ys)]
            gts.append({"image_id":image_id,"category_id":int(p[0]),"bbox":box,
                        "scale":get_scale((box[2]-box[0])*(box[3]-box[1]))})

    if result.boxes is None: continue
    for box,score,cls in zip(result.boxes.xyxy.cpu().numpy(),
                             result.boxes.conf.cpu().numpy(),
                             result.boxes.cls.cpu().numpy().astype(int)):
        x1,y1,x2,y2 = box
        preds.append({"image_id":image_id,"category_id":int(cls),"bbox":[x1,y1,x2,y2],
                      "score":float(score),"scale":get_scale((x2-x1)*(y2-y1))})

cats   = [{"id":i,"name":str(i)} for i in range(80)]
ann_id = 1

for scale in ["small","medium","large"]:
    sp = [p for p in preds if p["scale"]==scale]
    sg = [g for g in gts   if g["scale"]==scale]

    gt_json = {"images":[{"id":i} for i in set(x["image_id"] for x in sg+sp)],
               "categories":cats, "annotations":[]}
    for g in sg:
        x1,y1,x2,y2 = g["bbox"]
        gt_json["annotations"].append({"id":ann_id,"image_id":g["image_id"],
            "category_id":g["category_id"],"bbox":[x1,y1,x2-x1,y2-y1],
            "area":(x2-x1)*(y2-y1),"iscrowd":0})
        ann_id += 1

    pd_json = [{"image_id":p["image_id"],"category_id":p["category_id"],"score":float(p["score"]),
                "bbox":[float(p["bbox"][0]),float(p["bbox"][1]),float(p["bbox"][2]-p["bbox"][0]),float(p["bbox"][3]-p["bbox"][1])]}
               for p in sp]

    json.dump(gt_json, open(f"gt_{scale}.json","w"))
    json.dump(pd_json, open(f"pd_{scale}.json","w"))

    if sp and sg:   # guard against ZeroDivisionError in TIDE
        tide = TIDE()
        tide.evaluate_range(Data(f"gt_{scale}.json"), Data(f"pd_{scale}.json"), mode=TIDE.BOX)
        print(f"\n── TIDE: {scale} ──"); tide.summarize()
    else:
        print(f"\n── TIDE: {scale} — skipped (empty) ──")

    pb = defaultdict(list); gb = defaultdict(list)
    for p in sp: pb[p["image_id"]].append(p)
    for g in sg: gb[g["image_id"]].append(g)

    confs, correct = [], []
    for iid, ps in pb.items():
        matched = set()
        for p in sorted(ps, key=lambda x:-x["score"]):
            best,bi = 0,-1
            for i,g in enumerate(gb[iid]):
                if i in matched or g["category_id"]!=p["category_id"]: continue
                v = iou(p["bbox"],g["bbox"])
                if v>best: best,bi=v,i
            if best>=0.5: matched.add(bi)
            confs.append(p["score"]); correct.append(float(best>=0.5))

    if not confs:
        print(f"ECE ({scale}) = N/A"); continue
    confs,correct = np.array(confs),np.array(correct)
    ece = sum((mask:=(confs>lo)&(confs<=hi)).sum()/len(confs)*abs(correct[mask].mean()-confs[mask].mean())
              for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]) if mask.sum()>0)
    print(f"ECE ({scale}) = {ece:.4f}")