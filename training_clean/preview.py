import sys
sys.path.append("/Users/yogesh/c0DE/Signature-Verification-FIRST-/training")
import cv2, numpy as np, pandas as pd
from augment import SignatureAugment

IMG_ROOT = "/Users/yogesh/c0DE/Signature-Verification-FIRST-/training_clean"   # change this to your image folder
df = pd.read_csv("manifest_aug.csv")
row = df[df.split == "train"].sample(1).iloc[0]
img = cv2.imread(f"{IMG_ROOT}/{row.image_path}", cv2.IMREAD_GRAYSCALE)

aug = SignatureAugment()
tiles = [img] + [aug(img) for _ in range(5)]
grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])
cv2.imwrite("aug_preview.png", grid)
print("saved aug_preview.png for", row.image_path)
