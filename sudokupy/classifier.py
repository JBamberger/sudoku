import os
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np

PATH = './digit_classifier.pth'

patches = np.empty((9, 64, 64))
for i in range(9):
    patch = cv.imread(f'train_data/{i + 1}.jpg', cv.IMREAD_GRAYSCALE)
    _, threshed = cv.threshold(patch, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    patches[i, :, :] = threshed


def deskew(img):
    out_size = img.shape[0]
    moments = cv.moments(img)

    if abs(moments['mu02']) < 1e-2:
        return img.copy()

    skew = moments['mu11'] / moments['mu02']
    warp_mat = np.array([
        [1.0, skew, -0.5 * out_size * skew],
        [0.0, 1.0, 0.0]
    ], dtype=np.float32)

    img = cv.warpAffine(img, warp_mat, (out_size, out_size), flags=cv.WARP_INVERSE_MAP | cv.INTER_AREA)
    return img


def classify_digit(patch) -> int:
    _, threshed = cv.threshold(patch, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    xored = np.logical_not(np.logical_xor(patches, threshed))
    nnzs = np.count_nonzero(xored, axis=(1, 2))

    xored = np.concatenate([p.squeeze(0) for p in np.split(xored, 9, axis=0)], axis=1).astype(np.uint8) * 255
    templates = np.concatenate([p.squeeze(0) for p in np.split(patches, 9, axis=0)], axis=1)

    # cv.imshow('input', threshed)
    # cv.imshow('templates', templates)
    # cv.imshow('xored', xored)
    # cv.waitKey()


    return np.argmax(nnzs) + 1

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def load(self):
#         self.load_state_dict(torch.load(PATH))
#
#     def classify(self, patches):
#         # TODO: preprocess patches
#
#         patches = torch.from_numpy(patches).permute((2, 0, 1)).unsqueeze(0)
#
#         result = self[patches]
#
#         # TODO: postprocess patches
#
#         return result
#
#
# class DigitDataset(Dataset):
#
#     def __init__(self):
#         # 2000x1000
#
#         digits_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'share', 'digits.png')
#         digits = cv.imread(digits_path, cv.IMREAD_GRAYSCALE)
#         digits = torch.from_numpy(digits).view(1000, 2000).contiguous()
#         digits = digits.view(50, 20, 2000).contiguous().permute(2, 0, 1)
#         digits = digits.reshape(5000, 1, 20, 20)
#
#         cv.imshow('Image', digits[0, 0, :, :].numpy())
#         cv.waitKey()
#
#         return
#
#     def __getitem__(self, item):
#         return
#
#     def __len__(self):
#         return
#
#
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# trainset = DigitDataset()
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
#                                           shuffle=True, num_workers=2)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
#
# def train():
#     net = Net()
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#
#     for epoch in range(2):
#         for i, data in enumerate(trainloader, 0):
#             inputs, labels = data
#
#             optimizer.zero_grad()
#
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#     torch.save(net.state_dict(), PATH)
#
#
# if __name__ == '__main__':
#     # train()
#     pass
