import os
import cv2
import json

def read_label_and_image(label_path, img_prefix):
    with open(label_path,'r') as f:
        labels = json.load(f)
    imgpath = os.path.join(img_prefix, os.path.basename(label_path)).replace('.json', '.jpg')
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return labels, img