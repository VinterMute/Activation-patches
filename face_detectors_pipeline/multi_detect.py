import cv2
from mtcnn import MTCNN
import dlib
from tqdm import tqdm
import tensorflow as tf
from facenet_pytorch import MTCNN as pytorch_mtcnn
import numpy as np

# MTCNN initialization
mtcnn_torch = pytorch_mtcnn(keep_all=True, device="cpu")

detector_mtcnn = MTCNN()
detector_HOG = dlib.get_frontal_face_detector()
model_path = "mmod_human_face_detector.dat"
cnn_face_detector = dlib.cnn_face_detection_model_v1(model_path)


input_dir = "hog"
output_dir = "hog_check"


def add_gaussian_blur(image, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def add_noise(image, noise_percentage):
    mean = 0
    std_dev = np.sqrt(noise_percentage / 100) * 255
    gaussian_noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

import os
target_photos = os.listdir(input_dir)
for target in tqdm(target_photos):
    frame = cv2.imread(input_dir+"/"+target)
    # frame = add_gaussian_blur(frame, 3 )
    # frame = add_noise(frame, 1)

    # Face detection
    try:
        faces_mtcnn = detector_mtcnn.detect_faces(frame)
    except Exception:
        faces_mtcnn = []

    boxes_torch, _ = mtcnn_torch.detect(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_HOG = detector_HOG(gray)

    faces_CNN = cnn_face_detector(frame)

    for face in faces_mtcnn:
        x, y, width, height = face['box']
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 100, 255), 2)
        label = 'MTCNN'
        # Text color: orange
        text_color = (0, 140, 255)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    for face in faces_HOG:
        x, y, width, height = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        label = 'HOG'
        # Text color: green
        text_color = (50, 205, 50)
        cv2.putText(frame, label, (x + width - cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0], y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    if boxes_torch is not None:
        for face in boxes_torch:
            x1, y1, x2, y2 = map(int, face)
            width = x2 - x1
            height = y2 - y1
            cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 200, 60), 2)
            label = 'Torch_MTCNN'
            # Text color: light green
            text_color = (60, 220, 60)
            cv2.putText(frame, label, (x1, y1 + height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    for face in faces_CNN:
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        label = 'CNN'
        # Text color: red
        text_color = (255, 50, 50)
        cv2.putText(frame, label, (x + w - cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0], y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    cv2.imwrite(f"{output_dir}/{target}", frame)  # Save result

cv2.destroyAllWindows()
