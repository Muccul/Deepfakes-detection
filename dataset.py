import torch
import visdom
import csv
import os, glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import argparse

class Face(Dataset):
    def __init__(self, roots='', resize=299, mode='train', filename="All_c40_F"):
        super(Face, self).__init__()

        self.roots = roots
        self.resize = resize
        self.mode = mode
        self.name2label = {}
        filename = filename + "_" + mode

        for step, name in enumerate(roots):
            self.name2label[name.split("/")[-3]] = step

        self.images, self.labels = self.load_csv(filename + '.csv')

    def load_csv(self, filename):
        print("load csv：", filename)
        if not os.path.exists(os.path.join('./data/csv', filename)):
            images = []
            for face_class in self.roots:
                class_images = []
                class_name = os.listdir(face_class)
                class_name.sort()
                for step, face_imgaes in enumerate(class_name):
                    if self.mode == "train":
                        if step < 1000 * 0.7:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                    elif self.mode == "val":
                        if step >= 1000 * 0.7 and step < 1000 * 0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                    elif self.mode == "test":
                        if step >= 1000 * 0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                images += class_images

            with open(os.path.join('./data/csv', filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = re.split(r'[/\\]', img)[-5]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        images, labels = [], []
        with open(os.path.join('./data/csv', filename)) as f:
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
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
        ])

        img = tf(img)
        label = torch.tensor(label)

        return img, label


# python dataset.py --data Deepfakes --compression c40 --mode C
def main():
    parser = argparse.ArgumentParser(description="Face++ dataset")
    parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "All"], help="dataset consist of datas")
    parser.add_argument("--compression", type=str, default="c40", choices=["c23", "c40"])
    parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
    opt = parser.parse_args()

    ALL_DATA_ROOTS = {
        "Original" : "data/original_sequences/",
        "Deepfakes":"data/manipulated_sequences/Deepfakes/",
        "Face2Face":"data/manipulated_sequences/Face2Face/",
        "FaceSwap":"data/manipulated_sequences/FaceSwap/",
        "NeuralTextures":"data/manipulated_sequences/NeuralTextures/"
      }


    roots = []    # ["data/manipulated_sequences/Deepfakes/c23/crop_images", "data/original_sequences/c23/crop_images"]
    filename = opt.data + "_" + opt.compression + "_" + opt.mode

    for its in ALL_DATA_ROOTS.keys():
        ALL_DATA_ROOTS[its] = ALL_DATA_ROOTS[its] + opt.compression
        if opt.mode == "F":
            ALL_DATA_ROOTS[its] = ALL_DATA_ROOTS[its] + "/full_images"
        else:
            ALL_DATA_ROOTS[its] = ALL_DATA_ROOTS[its] + "/crop_images"

    if opt.data == "All":
        for its in ALL_DATA_ROOTS.values():
            roots.append(its)
    else:
        roots.append(ALL_DATA_ROOTS["Original"])
        roots.append(ALL_DATA_ROOTS[opt.data])

    train_dataset = Face(roots=roots, mode='train', filename=filename)
    val_dataset = Face(roots=roots, mode='val', filename=filename)
    test_dataset = Face(roots=roots, mode='test', filename=filename)

    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8)
    # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=8)

    # for step, (x, y) in enumerate(train_loader):
    #     print(step, x.shape)
    # train_loader = DataLoader(face_train, batch_size=64, shuffle=True, num_workers=16, pin_memory=True)


if __name__ == '__main__':
    main()

