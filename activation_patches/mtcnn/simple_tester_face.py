import torch
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os

# Initialize MTCNN
mtcnn = MTCNN(keep_all=True, device="cpu")

# Load image
img_path = 'test_h.jpg'
if not os.path.exists(img_path):
    print(f"The path {img_path} does not exist. Please check the file path and try again.")
else:
    img = Image.open(img_path)

    # Detect faces
    boxes, _ = mtcnn.detect(img)

    # Draw bounding box
    draw = ImageDraw.Draw(img)
    if boxes is not None:
        for box in boxes[0:1]:
            draw.rectangle(box.tolist(), outline=(0, 255, 0), width=5)


    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()