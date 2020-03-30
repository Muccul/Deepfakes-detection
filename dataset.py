import torch
import visdom
import csv, random
import os, glob
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re

# ["data/manipulated_sequences/Deepfakes/c23/crop_images", "data/original_sequences/c23/crop_images"]
class Face(Dataset):
    def __init__(self, roots, resize=299, mode='train', filename="ori"):
        super(Face, self).__init__()

        self.roots = roots
        self.resize = resize
        self.mode = mode

        self.name2label = {}
        for step, name in enumerate(roots):
            self.name2label[name.split("/")[-3]] = step

        self.images, self.labels = self.load_csv('face_' + filename + "_" +  mode + '.csv')


    def load_csv(self, filename):
        print("load csv：", filename)
        if not os.path.exists(os.path.join('./', filename)):
            images = []
            for face_class in self.roots:
                class_images = []
                class_name = os.listdir(face_class)
                class_name.sort()
                for step, face_imgaes in enumerate(class_name):
                    if self.mode == "train":
                        if step < 1000*0.7:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                    elif self.mode == "val":
                        if step >= 1000*0.7 and step < 1000*0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                    elif self.mode == "test":
                        if step >= 1000*0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                images += class_images

            with open(os.path.join('./', filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = re.split(r'[/\\]', img)[-5]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [], []
        with open(os.path.join('./', filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.ToTensor(),
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label


if __name__ == '__main__':
    roots = ["data/manipulated_sequences/Deepfakes/c23/crop_images", "data/original_sequences/c23/crop_images"]
    train_dataset = Face(roots=roots, mode='train', filename="Deepfakes")
    val_dataset = Face(roots=roots, mode='val', filename="Deepfakes")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    for step, (x, y) in enumerate(train_loader):
        print(step, x.shape)
    # train_loader = DataLoader(face_train, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)