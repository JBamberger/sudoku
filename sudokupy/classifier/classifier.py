import os
import random

import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt

import config

patches = np.empty((9, 64, 64))
for i in range(9):
    patch = cv.imread(os.path.join(config.digit_samples_path, f'{i + 1}.jpg'), cv.IMREAD_GRAYSCALE)
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

    # xored = np.concatenate([p.squeeze(0) for p in np.split(xored, 9, axis=0)], axis=1).astype(np.uint8) * 255
    # templates = np.concatenate([p.squeeze(0) for p in np.split(patches, 9, axis=0)], axis=1)
    # cv.imshow('input', threshed)
    # cv.imshow('templates', templates)
    # cv.imshow('xored', xored)
    # cv.waitKey()

    return np.argmax(nnzs) + 1


def prep_cell(x):
    x = cv.resize(x, (20, 20), interpolation=cv.INTER_AREA)
    _, threshed = cv.threshold(x, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    return threshed


inference_transform = transforms.Compose([
    # transforms.Lambda(prep_cell),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])


class Net(nn.Module):
    # def __init__(self, exclude_zero=True):
    #     super(Net, self).__init__()
    #     self.conv1 = nn.Conv2d(1, 6, 5)
    #     self.pool = nn.MaxPool2d(2, 2)
    #     self.conv2 = nn.Conv2d(6, 16, 5)
    #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
    #     self.fc2 = nn.Linear(120, 84)
    #     self.fc3 = nn.Linear(84, 9 if exclude_zero else 10)
    #
    # def forward(self, x):
    #     x = self.pool(F.relu(self.conv1(x)))
    #     x = self.pool(F.relu(self.conv2(x)))
    #     x = x.view(-1, 16 * 5 * 5)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x

    def __init__(self, exclude_zero=True, size=20):
        flat_dim = ((size - 4) // 2) ** 2 * 64

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(flat_dim, 128)
        self.fc2 = nn.Linear(128, 9 if exclude_zero else 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def load(self, path=config.classifier_checkpoint_path):
        self.load_state_dict(torch.load(path))

    def save(self, path=config.classifier_checkpoint_path):
        torch.save(self.state_dict(), path)

    def classify(self, patches):
        patches = inference_transform(patches)
        patches = patches.unsqueeze(0)
        result = self(patches)

        # TODO: postprocess patches

        result = torch.argmax(result, dim=1)

        return result + 1


class MyDigitDataset(Dataset):
    def __init__(self, transform=None, path=None, split_idx=-30):
        self.transform = transform
        if path is None:
            path = config.digit_dataset_path

        digits = []
        labels = []
        for i in range(1, 10):
            digit_path = os.path.join(path, str(i))
            img_names = os.listdir(digit_path)
            random.shuffle(img_names)
            if split_idx >= 0:
                img_names = img_names[:split_idx]
            else:
                img_names = img_names[abs(split_idx):]

            for image_name in img_names:
                img_path = os.path.join(digit_path, image_name)
                img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
                digits.append(img)
                labels.append(i - 1)

        self.digits = np.array(digits)
        self.labels = torch.tensor(labels)

    def __getitem__(self, item):
        if self.transform is not None:
            return transform(self.digits[item]), self.labels[item]
        return np.expand_dims(self.digits[item], axis=-1), self.labels[item]

    def __len__(self):
        return self.labels.numel()


class DigitDataset(Dataset):

    def __init__(self, exclude_zero=True, transform=None, split_idx=500):
        self.transform = transform

        # 2000x1000
        digits = cv.imread(config.digit5k_dataset_path, cv.IMREAD_GRAYSCALE)

        # shape [1000, 2000]
        # digits = torch.from_numpy(digits)

        digits = digits.reshape(50, 20, 100, 20)
        digits = digits.transpose(0, 2, 1, 3)
        digits = digits.reshape(10, 500, 20, 20)

        if exclude_zero:
            digits = digits[1:, ...]

        if split_idx >= 0:
            digits = digits[:, :split_idx, :, :]
        else:
            digits = digits[:, abs(split_idx):, :, :]

        samples_per_digit = digits.shape[1]
        self.digits = digits.reshape(-1, 20, 20, 1)
        self.classes = torch.arange(9 if exclude_zero else 10).repeat_interleave(samples_per_digit)

    def __getitem__(self, item):
        if self.transform is not None:
            return self.transform(self.digits[item, :, :, :]), self.classes[item]
        else:
            return self.digits[item, :, :, :], self.classes[item]

    def __len__(self):
        return self.classes.numel()


class Average:
    def __init__(self):
        self.sum = 0
        self.avg = 0
        self.count = 0

    def update(self, value, size=1):
        self.sum += value
        self.count += size
        self.avg = self.sum / self.count

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.count = 0


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])


def train(use_cuda=True):
    # split_idx = 400
    # trainset = DigitDataset(transform=transform, split_idx=split_idx)
    # valset = DigitDataset(transform=transform, split_idx=-split_idx)

    split_idx = 30
    trainset = MyDigitDataset(transform=transform, split_idx=-split_idx)
    valset = MyDigitDataset(transform=transform, split_idx=split_idx)
    print(f'Trainset size: {len(trainset)} Valset size: {len(valset)}.')

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    net = Net(size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if use_cuda:
        net.cuda()
        criterion.cuda()

    train_losses = []
    val_losses = []
    for epoch in range(10):

        train_loss = Average()
        train_accuracy = Average()
        net.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data

            if use_cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            accuracy = torch.sum(outputs.argmax(dim=1) == labels).item()
            train_accuracy.update(accuracy, labels.numel())
            train_loss.update(loss.item(), labels.numel())
            loss.backward()
            optimizer.step()

        train_losses.append(train_loss.avg)

        val_loss = Average()
        val_accuracy = Average()
        with torch.no_grad():
            net.eval()
            for i, data in enumerate(valloader):
                inputs, labels = data
                if use_cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = net(inputs)
                accuracy = torch.sum(outputs.argmax(dim=1) == labels).item()
                val_accuracy.update(accuracy, labels.numel())
                loss = criterion(outputs, labels)
                val_loss.update(loss, labels.numel())
        print(
            f'{epoch}: Train loss: {train_loss.avg:f} Validation loss: {val_loss.avg:f} Train Accuracy: {train_accuracy.avg:f} Val Accuracy: {val_accuracy.avg:f}')
        val_losses.append(val_loss.avg)

    plt.plot(train_losses, label='Train_losses')
    plt.plot(val_losses, label='Train_losses')
    plt.legend()
    plt.show()

    net.save()


if __name__ == '__main__':
    train()
