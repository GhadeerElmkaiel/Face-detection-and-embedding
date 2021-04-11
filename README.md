# Face Detection and Embedding

## Description

The main task is to detect faces in images and get embedding for each face. To achieve this we need to do the following steps.

- Detect faces in the image using a pretrained model.
- Aligne the faces.
- Use pretrained model to give an embedding vector for each face.

the used models are:

- Face detection neural network [facenet-pytorch](https://github.com/timesler/facenet-pytorch/blob/master/examples/face_tracking.ipynb)
- Faces embedding neural network (encoder) [Pytorch InsightFace](https://github.com/nizhib/pytorch-insightface)

## Installation

**facenet-pytorch** can be installed using pip:

```bash
pip install facenet-pytorch
```

To install **InsightFace** see the [original repository](https://github.com/nizhib/pytorch-insightface)

## Code

importing used models:

```python
from facenet_pytorch import MTCNN
import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank
import matplotlib.pyplot as plt
```

Prepare the **mtcnn** neural-network and set it with cuda if available:

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Creating a face detection pipeline using MTCNN:
mtcnn = MTCNN(keep_all=True, device=device)

# Read testing image
img = Image.open("img1.jpg")
```

getting the surrounding boxes and face's landmarks:

```python
# Get boxes around each face and landmarks for each face.
boxes, probs ,landmarks = mtcnn.detect(img, landmarks=True)
```

For visualizing the results

```python
# Visualizing the boxes and landmarks
img_draw = img.copy()
draw = ImageDraw.Draw(img_draw)
for i, box in enumerate(boxes):
    draw.rectangle(box.tolist(), outline=(255,0,0), width=6)
    draw.point(landmarks[i])
img_draw.show()
```

Using the MTCNN model returns not only surrounding box, but also five face landmarks which can be used to alligne the face in the image for further face embeddings.  
To do the allignment we need to define the position of the five face landmarks in a (112X112) image "which is the size used in face embedding".  
We define the positions of the fve landmarks as follows:

```python

# Defining the position of face's landmarks in an image of size (112X112)
REFERENCE_FACIAL_POINTS = np.array([
    [35.3437  ,  51.69630051],
    [76.453766,  51.50139999],
    [56.0294  ,  71.73660278],
    [39.14085 ,  92.3655014 ],
    [73.18488 ,  92.20410156]
], np.float32)

```

We define a function which find the transformation matrix to be used in transforming the original face landmarks to the needed positions in a (112X112) image.  
for only two points it is possible to find a transformation that satisfy the wanted transformation exactly, but in the case of more points, it is not guaranteed to find a transformation that satisfies the wanted transformations.
In this case we need to find the transformation which causes the minimum accumulated error for all points.

```python
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

```

For each face we need to apply a transformation.
We use the function to get the transformation for the first face in the image

```python
# Getting the transformation
similar_trans_matrix = findNonreflectiveSimilarity(np.array(landmarks[0]).astype(np.float32), REFERENCE_FACIAL_POINTS)
```

Then we apply the transformation on a copy of the image

```python
# applying the transformation on the landmarks
img2 = img.copy()
aligned_face = cv2.warpAffine(src=np.array(img2), M=similar_trans_matrix, dsize=(112, 112))
```

To Test the algorithm we show the Face after applying the first transformation and showing the landmarks. 

```python
# Testing the result for one of the faces on the Image.
test_image = Image.fromarray(aligned_face)
draw = ImageDraw.Draw(test_image)
for i in range(len(REFERENCE_FACIAL_POINTS)):
    draw.point(REFERENCE_FACIAL_POINTS[i])

plt.figure(figsize=(5, 5))
plt.imshow(test_image)
plt.show()
```

Now we apply the same for each face in the image and store them in an array.  

```python
# Getting an array of cropped and transformed image for each face.
faces = []
for i in range(len(landmarks)):
    similar_trans_matrix = findNonreflectiveSimilarity(np.array(landmarks[i]).astype(np.float32), REFERENCE_FACIAL_POINTS)
    img_copy = img.copy()
    aligned_face = cv2.warpAffine(src=np.array(img_copy), M=similar_trans_matrix, dsize=(112, 112))
    faces.append(aligned_face)
    plt.imshow(aligned_face)
    plt.show()
```

we define the face embedder using **insightface** pretrained neural network.

```python
# Defining the face embedder as a pretrained model
embedder = insightface.iresnet100(pretrained=True)
embedder.eval()
```

We also need to define a preprocessing transformation for each image before applying the face embedding step.

```python
# Defining a preprocess for each face pefore passing it to the network.
mean = [0.5] * 3
std = [0.5 * 256 / 255] * 3
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

Now we get the features for all faces.

```python
# Getting the embedding features for each face 
features_all = []
with torch.no_grad():
    for face in faces:
        tensor = preprocess(face)
        feature_face = embedder(tensor.unsqueeze(0))
        features_all.append(feature_face)
```





