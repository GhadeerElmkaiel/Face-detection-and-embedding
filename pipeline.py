from facenet_pytorch import MTCNN
import insightface
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import matplotlib.pyplot as plt




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Creating a face detection pipeline using MTCNN:
mtcnn = MTCNN(keep_all=True, device=device)

# Read testing image
img = Image.open("img1.jpg")

# Get boxes around each face and landmarks for each face.
boxes, probs ,landmarks = mtcnn.detect(img, landmarks=True)

# Visualizing the boxes and landmarks
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, box in enumerate(boxes):
    draw.rectangle(box.tolist(), outline=(255,0,0), width=6)
    draw.point(landmarks[i])

img_draw.show()

# Defining the position of face's landmarks in an image of size (112X112)
REFERENCE_FACIAL_POINTS = np.array([
    [35.3437  ,  51.69630051],
    [76.453766,  51.50139999],
    [56.0294  ,  71.73660278],
    [39.14085 ,  92.3655014 ],
    [73.18488 ,  92.20410156]
], np.float32)

def findNonreflectiveSimilarity(uv, xy, K=2):

    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))  # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))  # use reshape to keep a column vector
    v = uv[:, 1].reshape((-1, 1))  # use reshape to keep a column vector
    U = np.vstack((u, v))

    # We know that X * r = U
    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U)
        r = np.squeeze(r)
    else:
        raise Exception('cp2tform:twoUniquePointsReq')

    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss,  sc, 0],
        [tx,  ty, 1]
    ])


    T = inv(Tinv)

    T[:, 2] = np.array([0, 0, 1])

    T = T[:, 0:2].T

    return T

# Getting the transformation
similar_trans_matrix = findNonreflectiveSimilarity(np.array(landmarks[7]).astype(np.float32), REFERENCE_FACIAL_POINTS)

# applying the transformation on the landmarks
img2 = img.copy()
aligned_face = cv2.warpAffine(src=np.array(img2), M=similar_trans_matrix, dsize=(112, 112))

# Testing the result for one of the faces on the Image.
test_image = Image.fromarray(aligned_face)
draw = ImageDraw.Draw(test_image)
for i in range(len(REFERENCE_FACIAL_POINTS)):
    draw.point(REFERENCE_FACIAL_POINTS[i])

plt.figure(figsize=(5, 5))
plt.imshow(test_image)
plt.show()

# Getting an array of cropped and transformed image for each face.
faces = []
for i in range(len(landmarks)):
    similar_trans_matrix = findNonreflectiveSimilarity(np.array(landmarks[i]).astype(np.float32), REFERENCE_FACIAL_POINTS)
    img_copy = img.copy()
    aligned_face = cv2.warpAffine(src=np.array(img_copy), M=similar_trans_matrix, dsize=(112, 112))
    faces.append(aligned_face)
    plt.imshow(aligned_face)
    plt.show()

# Defining the face embedder as a pretrained model
embedder = insightface.iresnet100(pretrained=True)
embedder.eval()

# Defining a preprocess for each face pefore passing it to the network.
mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Getting the embedding features for each face 
features_all = []
with torch.no_grad():
    for face in faces:
        tensor = preprocess(face)
        feature_face = embedder(tensor.unsqueeze(0))
        features_all.append(feature_face)

