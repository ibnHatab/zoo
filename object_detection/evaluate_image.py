

import os
import sys
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import torch

# %cd object_detection
try: sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
except: pass
sys.path.append('../src')

from model import get_instance_segmentation_model
from data import get_transform
from utils import load_checkpoint

sys.argv[1] = 'tv_image05.png'
img_path = sys.argv[1]
num_classes = 2

model = get_instance_segmentation_model(num_classes)

ckpt = load_checkpoint('ckpt_9.pth')
model.load_state_dict(ckpt['net'])

model.eval()

pillow_image = Image.open(img_path).convert('RGB')
transform = get_transform(train=False)
transformed_image = transform(pillow_image, None)[0]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
transformed_image = transformed_image.to(device)

prediction = model([transformed_image])
if len(prediction) > 0:
    print(f"Objects detected: {len(prediction[0]['boxes'])}")
    imgd = ImageDraw.Draw(pillow_image)
    for box in prediction[0]["boxes"]:
        imgd.rectangle(box.tolist(), outline='red', width=3)
    plt.imshow(pillow_image); plt.show() #
else:
    print("No objects detected")
    plt.imshow(pillow_image); plt.show() #
