from torch.nn.functional import interpolate
from random import random
import os
import dlib
import cv2
from matplotlib import pyplot as plt
from IPython.display import display, Image
#
# # Upload an image using OpenCV||
# img = cv2.imread('test.jpg')
#
#
# # Initialize face detector mmod
# detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
#
# # Find faces in the image
# detections = detector(img)
#
# for face in detections:
#     l, t, r, b = (face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom())
#     cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
#
# # Show image with faces outlined
#
# cv2.imwrite("detect.jpg", img)
#

from matplotlib import pyplot as plt
from  webcam_example import *
import torch.optim as optim
import sys
import time
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable
import numpy as np
import pickle
from skimage.transform import rotate
from torchvision import transforms
from skimage.feature import peak_local_max
from skimage.transform import resize as sk_resize


from PIL import Image


def random_shift_rotate(image, shift_max=20, angle_max=45):
    image_reshaped = image[0].transpose(1, 2, 0)

    # Generating a random rotation angle and shift
    angle = np.random.randint(-angle_max, angle_max + 1)
    shift_x = np.random.randint(-shift_max, shift_max + 1)
    shift_y = np.random.randint(-shift_max, shift_max + 1)

    rows, cols, _ = image_reshaped.shape

    # Create a rotation matrix and apply rotation
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    rotated_image = cv2.warpAffine(image_reshaped, rotation_matrix, (cols, rows))

    # apply shift
    shift_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_rotated_image = cv2.warpAffine(rotated_image, shift_matrix, (cols, rows))

    transformed_tensor = shifted_rotated_image.transpose(2, 0, 1).reshape(1, 3, 150, 150)

    return transformed_tensor

def validate_torch(img_path):
    # Code for using the model loaded into torch
    img = cv2.imread(img_path)

    model = get_model("dlib_facedetector_pytorch/face.xml")
    recepive_field_size, offset = recepive_field(model)
    # print "Receptive field", recepive_field_size, "Center offset", offset
    scales = [2.7, 3.5]
    coordinates = []

    t0 = time.time()
    coordinates = detect_multi_scale(model, img, scales, recepive_field_size)
    t1 = time.time()
    # print("Inference time for frame", t1 - t0)


    for x, y, width, height, score in coordinates:
        x = int(x - offset//2)
        y = int(y - offset//2)
        cv2.rectangle(img, (x-width//2, y-height//2), (x+width//2, y+height//2), (0, 255, 0))
    if len(coordinates) != 0:
        cv2.imwrite("detect_torch.jpg", img)
        print("*** Find Face ***")
        return True
    else:
        print("Face not found")
        return False

def validate_face_rec(image_path):
    #Validare face recognition
    import face_recognition
    # Upload your image
    image_path= f'test/frame{i:04}.jpg'
    unknown_image = face_recognition.load_image_file(image_path)

    # Find all the faces in the image
    face_locations = face_recognition.face_locations(unknown_image,model="cnn")
    print(face_locations)
    if len(face_locations) > 0:
        print("*** find face face_recognition ** ")
        return True
    else:
        return False

def img_CV_norm(img):
    r = 122.781998
    g = 117.000999
    b = 104.297997

    img[:, :, 0] -= b
    img[:, :, 1] -= g
    img[:, :, 2] -= r

    img = img[:, :, ::-1].copy() / 256.0

    img = img.transpose((2, 0, 1))
    img = np.float32(img)
    img = torch.from_numpy(img)

    img = img.unsqueeze(0)
    return img


def get_reference_output(model,img):
    model.eval()
    img = np.float32(img)
    img = img_CV_norm(img)
    # output = model(Variable(img, volatile=True))
    with torch.no_grad():
        output = model(img)
    # output = np.float32(output2)
    #Save for using something else 
    with open("refer_output.pickle", "wb") as f:
        pickle.dump(output, f)

    # output = output.data.numpy()
    # output = output[0, 0, :, :]
    # output = sk_resize(output, output_shape=img.shape[:2], preserve_range=True)
    # output = np.float32(output)
    # coordinates = peak_local_max(output, min_distance=30, threshold_abs=-1)
    # # output = model(Variable(img_CV_norm(img), volatile=True))
    # output = output.data.numpy()
    # # output = output[0, 0, :, :]
    return output


def tensor_to_np_image(input):
    img = input.cpu().data.numpy()[0, :]
    img = np.transpose(img, (1, 2, 0))
    img = img - np.min(img)
    img = img / np.max(img)
    img = img[:, :, ::-1]
    return img
def tensor_to_img_CV(tensor):
    r = 122.781998
    g = 117.000999
    b = 104.297997

    # Convert a PyTorch tensor to a numpy array
    img = tensor.squeeze(0).cpu().numpy()

    
    img = img.transpose((1, 2, 0))

    # We do the reverse conversion (multiply by 256 and invert the channels)
    img = img * 256.0
    img = img[:, :, ::-1]

    # Restoring original pixel values
    img[:, :, 0] += b
    img[:, :, 1] += g
    img[:, :, 2] += r

    # Let's make sure the pixel values are in the correct range
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img



def rotate_img(img, angle):
    for index in range(3):
        channel = img[0, index, :, :]
        channel = rotate(channel, angle, resize=False, preserve_range=True)
        img[0, index, :, :] = channel
    return np.float32(img)

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

if __name__ == "__main__":
    img = cv2.imread('test.jpg')
    model = get_model("dlib_facedetector_pytorch/face.xml")
    model.eval()
    reference_tensor = get_reference_output(model, img)

    N = 150 # patch size

    # input = np.float32(np.ones((1, 3, N, N))) / 255
    input = np.float32(np.random.rand(1, 3, N, N)) / 255
    input = Variable(torch.from_numpy(input), requires_grad=True)


    optimizer = optim.Adam([input], lr=0.01)
    criterion = torch.nn.MSELoss()

    iterations = 1000
    lr=1

    loss_plt = []
    for i in range(iterations):

        # img = input.data.numpy()

        model.zero_grad()
        out = model(input)
        loss = criterion(out, reference_tensor)
        size = out.size(2)
        # loss = out[0, 0, size // 2, size // 2]
        loss.backward()

        # optimizer.step()

        grad_cpu = input.grad
        grad_cpu = grad_cpu / torch.norm(grad_cpu, 2)
        data = grad_cpu.data.numpy()

        input = input.clone().detach() - lr * grad_cpu
        input.requires_grad = True
        tensor_to_np_image(input)
        print(f"Iteration {i}, Loss: {loss.item()}")
        loss_plt.append(loss.item())


        # angle = np.float32(np.random.uniform(-30, 30)) * 1
        #
        img = input.data.numpy()
        # img = rotate_img(img, -angle)

        # use_flip = random() > 0.5
        # if use_flip:
        #     img[:, :, :, :] = img[:, :, :, ::-1]

        # transformed_tensor = random_shift_rotate(img,shift_max=3,angle_max=10)
        transformed_tensor = img
        input.data = torch.from_numpy(transformed_tensor)

        img = tensor_to_np_image(input)
        img = np.uint8(255 * img)
        cv2.imwrite(f'step-by-step_opt/frame{i:04}.jpg', img)


        #Stop optimization after dlib found face
        if validate_face_rec(f'step-by-step_opt/frame{i:04}.jpg') and validate_torch(f'step-by-step_opt/frame{i:04}.jpg'):
            print("Face Found, break optimizations")
            break

# os.system("ffmpeg -i  test/frame%04d.jpg test.gif -y")



