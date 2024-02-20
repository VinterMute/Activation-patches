import torch
from facenet_pytorch import MTCNN
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
from torch.nn.functional import interpolate
import  pickle

#First stage load MTCNN modeles

def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode="area")
    return im_data

def tensor_to_img(tensor, filename = f'output/test.jpg', save = True ):
    tensor = tensor[0]
    image = tensor.permute(1, 2, 0).detach().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    if save:
        img = np.uint8(255 * image )

        cv2.imwrite(filename, img)

    plt.imshow(image)
    plt.show()

def pad(boxes, w, h):
    # boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, 0]
    y = boxes[:, 1]
    ex = boxes[:, 2]
    ey = boxes[:, 3]

    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h

    return y, ey, x, ex
mtcnn = MTCNN(select_largest=False, keep_all=True, post_process=False)
loss_pnet = torch.nn.MSELoss()
loss_rnet = torch.nn.MSELoss()

loss_onet = torch.nn.MSELoss()
loss_onet_landmarks = torch.nn.MSELoss()

N = 512

# input = torch.randn(1, 3, N, N, requires_grad=True)
input = torch.ones(1, 3, N, N, requires_grad=True)
# input = torch.zeros(1, 3, N, N, requires_grad=True)
n_epochs = 20



for param in mtcnn.pnet.parameters():
    param.requires_grad = False

for param in mtcnn.rnet.parameters():
    param.requires_grad = False

for param in mtcnn.onet.parameters():
    param.requires_grad = False

lr = 1


pbar = tqdm(range(n_epochs))

#Read target outputs
with open("pnet_reflection_new.pickle", "rb") as f: #Круто что предусмотрел
    existing_data = pickle.load(f)


scales = existing_data["scales"]
outputs_model = existing_data["probs_pnet"]
rnet_boxes = existing_data["result_box"]
image_inds = existing_data["image_inds"]
old_boxes = existing_data["boxes_old"]
target_landmarks = existing_data["result_points"]


history_loss_pnet =[[] for _ in range(len(scales))]
history_loss_rnet= []
history_loss_onet = []

global_history_loss = []
for i in pbar:
    #P net optimization step
    for count, scale in enumerate(scales):
        input_resampled = imresample(input, (int(N * scale + 1), int(N * scale + 1)))
        reg, probs = mtcnn.pnet(input_resampled)
        iter_pnet_loss = loss_pnet(probs[:, 1], outputs_model[count][:, 1])
        history_loss_pnet[count].append(float(iter_pnet_loss))
        pbar.set_description(f"Loss {scale} - {history_loss_pnet[-1]}")
        iter_pnet_loss.backward()
        # Normalizate gradients
        grad_cpu = input.grad
        grad_cpu = grad_cpu / torch.norm(grad_cpu, 2)
        lr = 2
        input = input.clone().detach() - lr * grad_cpu
        input.requires_grad = True

    #Rnet optimization step
    # y, ey, x, ex = pad(rnet_boxes, N,N)

    y1, x1, y2, x2, _ = map(int, rnet_boxes[0])
    im_data = []
    im_crop = input[:, :, y1:y2, x1:x2]

    im_data.append(imresample(im_crop, (24, 24)))
    im_data = torch.cat(im_data, dim=0)

    rnet_out = mtcnn.rnet(im_data)
    out0 = rnet_out[0].permute(1, 0)
    out1 = rnet_out[1].permute(1, 0)
    score = out1[1, :]

    # Loss R-net
    iter_rnet_loss = loss_rnet(score, torch.ones_like(score))

    global_history_loss.append(history_loss_pnet)
    history_loss_rnet.append(float(iter_rnet_loss))
    iter_rnet_loss.backward()
    grad_cpu = input.grad
    grad_cpu = grad_cpu / torch.norm(grad_cpu, 2)
    lr= 1
    input = input.clone().detach() - lr * grad_cpu
    input.requires_grad = True


    #O-net attack
    im_data = []
    im_crop = input[:, :, y1:y2, x1:x2]
    im_data.append(imresample(im_crop, (48, 48)))
    im_data = torch.cat(im_data, dim=0)
    onet_out = mtcnn.onet(im_data)

    out0 = onet_out[0].permute(1, 0)
    out1 = onet_out[1].permute(1, 0)  # keypoints
    out2 = onet_out[2].permute(1, 0)  # score
    score = out2[1, :]

    iter_onet_loss = loss_onet(score, torch.ones_like(score))
    iter_onet_landmark = loss_onet_landmarks(out1,target_landmarks[0])
    history_loss_onet.append(float(iter_onet_loss))
    iter_onet_loss.backward(retain_graph=True)
    iter_onet_landmark.backward()


    grad_cpu = input.grad
    grad_cpu = grad_cpu / torch.norm(grad_cpu, 2)
    lr = 0.2
    input = input.clone().detach() - lr * grad_cpu
    input.requires_grad = True




tensor_to_img(input, save=True)


for i, param_errors in enumerate(history_loss_pnet):
    plt.plot(param_errors, label=f'{scales[i]}')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('P_R_nets attack')
plt.plot(history_loss_rnet,label=f'R-net loss',linestyle = '--',color="black")
plt.plot(history_loss_onet,label=f'O-net loss',linestyle = ':',color="green")

plt.legend()
plt.show()

