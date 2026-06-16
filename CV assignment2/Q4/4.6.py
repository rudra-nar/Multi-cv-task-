import cv2 as cv
import os
import numpy as np
from sklearn.cluster import KMeans

path      = r"C:\Users\Asus\Documents\CV assignment2"
data_path = os.path.join(path, "panorama_dataset")

image_files = sorted(f for f in os.listdir(data_path) if f.endswith(".png"))
images      = [cv.imread(os.path.join(data_path, f)) for f in image_files]

features = []
for img in images:
    hist = []
    for ch in range(3):
        h = cv.calcHist([img], [ch], None, [10], [0, 256])
        hist.extend(cv.normalize(h, h).flatten())
    features.append(hist)

labels   = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(np.array(features, dtype=np.float32))
clusters = {0: [], 1: [], 2: []}
for idx, lbl in enumerate(labels):
    clusters[lbl].append(images[idx])

for cid, imgs in clusters.items():

    sift  = cv.SIFT_create()
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))
    descs = [sift.detectAndCompute(cv.cvtColor(img, cv.COLOR_BGR2GRAY), None)[1] for img in imgs]

    n   = len(imgs)
    sim = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if descs[i] is not None and descs[j] is not None:
                raw = flann.knnMatch(descs[i], descs[j], k=2)
                c   = sum(1 for m, nxt in raw if m.distance < 0.7 * nxt.distance)
                sim[i][j] = sim[j][i] = c

    visited = {int(np.argmax(sim.sum(axis=1)))}
    order   = list(visited)
    while len(order) < n:
        cur = order[-1]
        nxt = max((j for j in range(n) if j not in visited), key=lambda j: sim[cur][j])
        order.append(nxt); visited.add(nxt)

    imgs = [imgs[i] for i in order]
    print(f"  Stitching order: {order}")

    pano = imgs[0]
    for i in range(1, len(imgs)):
        print(f"  image {i+1}/{len(imgs)} ...")

        kp1, desc1 = sift.detectAndCompute(cv.cvtColor(pano,    cv.COLOR_BGR2GRAY), None)
        kp2, desc2 = sift.detectAndCompute(cv.cvtColor(imgs[i], cv.COLOR_BGR2GRAY), None)

        raw  = flann.knnMatch(desc2, desc1, k=2)
        good = [m for m, nxt in raw if m.distance < 0.7 * nxt.distance]

        if len(good) < 10:
            print("  [WARN] not enough matches, skipping"); continue

        src = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src, dst, cv.RANSAC, 4.0)

        if M is None or mask.sum() < 8:
            print("  [WARN] bad homography, skipping"); continue

        h1, w1 = pano.shape[:2]
        h2, w2 = imgs[i].shape[:2]
        corners_new  = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
        corners_base = np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2)
        all_corners  = np.concatenate([cv.perspectiveTransform(corners_new, M), corners_base])
        x_min = int(all_corners[:,0,0].min()) - 20
        y_min = int(all_corners[:,0,1].min()) - 20
        x_max = int(all_corners[:,0,0].max()) + 20
        y_max = int(all_corners[:,0,1].max()) + 20
        tx, ty = -x_min, -y_min
        size   = (x_max - x_min, y_max - y_min)
        offset = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

        base_w = cv.warpPerspective(pano,    offset,     size)
        new_w  = cv.warpPerspective(imgs[i], offset @ M, size)
        pano   = np.where(base_w == 0, new_w, base_w)

    _, mask = cv.threshold(cv.cvtColor(pano, cv.COLOR_BGR2GRAY), 1, 255, cv.THRESH_BINARY)
    x, y, w, h = cv.boundingRect(cv.findNonZero(mask))
    pano = pano[y:y+h, x:x+w]

    cv.imshow(f"Panorama {cid+1}", pano)
    cv.imwrite(os.path.join(path, f"panorama_{cid+1}.png"), pano)

cv.waitKey(0)