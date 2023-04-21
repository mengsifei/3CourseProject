from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import torch
import time
import torch.nn as nn
import math
from torchvision import models
import numpy as np
import os
import torchvision.transforms as transforms
from PIL import Image

def bn_relu(inplanes):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True))

def bn_relu_pool(inplanes, kernel_size=3, stride=2):
    return nn.Sequential(nn.BatchNorm2d(inplanes), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=kernel_size, stride=stride))

class AlexNet(nn.Module):
    def __init__(self, num_classes=1):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, bias=False)
        self.relu_pool1 = bn_relu_pool(inplanes=96)
        self.conv2 = nn.Conv2d(96, 192, kernel_size=5, padding=2, groups=2, bias=False)
        self.relu_pool2 = bn_relu_pool(inplanes=192)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu3 = bn_relu(inplanes=384)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu4 = bn_relu(inplanes=384)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2, bias=False)
        self.relu_pool5 = bn_relu_pool(inplanes=256)
        # classifier
        self.conv6 = nn.Conv2d(256, 256, kernel_size=5, groups=2, bias=False)
        self.relu6 = bn_relu(inplanes=256)
        self.conv7 = nn.Conv2d(256, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu_pool1(x)
        x = self.conv2(x)
        x = self.relu_pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu_pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = x.view(x.size(0), -1)
        return x


def load_model(pretrained_dict, new):
    model_dict = new.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    new.load_state_dict(model_dict)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])


def predict_img(img, model):
    model.eval()
    with torch.no_grad():
        # img = Image.open(path).convert('RGB')
        # print(type(img))
        img = transform(img)
        img = img.unsqueeze(0)
        output = model(img)
        score = output.cpu()[0][0]
        return score

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


def estimate_blur(image: np.array, threshold: int = 100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_map = cv2.Laplacian(image, cv2.CV_64F)
    score = np.var(blur_map)
    return score, bool(score < threshold)


def load_efficientnet():
    efficientnet_v2 = models.efficientnet_v2_l()
    efficientnet_v2.classifier[1] = nn.Linear(1280, 1, bias=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    efficientnet_v2 = efficientnet_v2.to(device)
    checkpoint = torch.load('data/efficientnet_v2.pth', map_location=torch.device('cpu'))
    efficientnet_v2.load_state_dict(checkpoint['model_state_dict'])
    return efficientnet_v2


def load_alexnet():
    net = AlexNet()
    # load pretrained model
    load_model(torch.load('data/alexnet.pth', map_location=torch.device('cpu'), encoding='latin1'), net)
    return net


def load_myAlexnet():
    alexnet = models.alexnet()
    alexnet.classifier[6] = nn.Linear(in_features=4096, out_features=1, bias=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    alexnet = alexnet.to(device)
    checkpoint = torch.load('data/my_alexnet.pth', map_location=torch.device('cpu'))
    alexnet.load_state_dict(checkpoint['model_state_dict'])
    return alexnet


def my_app(video_path, num_frame=0, frame_size=1020, box_size=800):
    start_time = time.time()
    model = load_myAlexnet()
    # model = load_efficientnet()
    # model = load_alexnet()
    cap = cv2.VideoCapture(video_path)
    if num_frame == 0:
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = 0
    outputs = []
    is_first = True
    threshold = 0
    while True:
        ret, frame = cap.read()
        count += 1
        if ret:
            frame = imutils.resize(frame, width=400)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detect(gray, 0)
            if len(faces) == 0:
                continue
            score_blur, is_blur = estimate_blur(gray)
            score_blur /= 3000
            shape = predict(gray, faces[0])
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < thresh:
                continue
            pil_image = Image.fromarray(frame)
            score = predict_img(pil_image, model)
            overall_score = score_blur + score
            if is_first:
                threshold = overall_score
                is_first = False
            if overall_score >= threshold:
                outputs.append((overall_score, count, score))
            if count >= num_frame and len(outputs) != 0:
                break
        else:
            break
    # if first frame is the best
    cap.set(1, max(outputs)[1])
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = detect(gray, 0)[0]
    diff = min(gray.shape[1] - face.right(), face.left(), face.top(), gray.shape[0] - face.top())
    box_size = min(diff, box_size)
    crop_img = frame[max(0, face.top() - box_size): min(face.bottom() + box_size, frame.shape[0]),
               max(face.left() - box_size, 0): min(face.right() + box_size, frame.shape[1])]
    crop_img = imutils.resize(crop_img, width=min(frame_size, crop_img.shape[1]))
    print("--- %s seconds ---" % (time.time() - start_time))
    return crop_img, np.array(max(outputs)[2])


# path = 'data/videos/tae.mp4'
# outputs = my_app(path)
# cap = cv2.VideoCapture(path)
# print("Best frame:", max(outputs)[1])
# cap.set(1, max(outputs)[1])
# ret, frame = cap.read()
# pil_image = Image.fromarray(frame)
# cv2.imshow('frame', frame)
# os.chdir('data/frame')
# filename = 'tae.jpg'
# cv2.imwrite(filename, frame)

