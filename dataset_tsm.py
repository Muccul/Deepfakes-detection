import torch
import visdom
import csv
import os, glob
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import argparse

# python dataset_tsm.py --data Face2Face --compression c40 --mode C
class Face(Dataset):
    def __init__(self, roots='', resize=299, mode='train', filename="All_c40_C"):
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
                            class_images.sort()
                    elif self.mode == "val":
                        if step >= 1000 * 0.7 and step < 1000 * 0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                            class_images.sort()
                    elif self.mode == "test":
                        if step >= 1000 * 0.85:
                            class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
                            class_images.sort()
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
            segment_imgs, segment_labs = [], []
            fg = 0
            for row in reader:
                img, label = row
                label = int(label)
                if fg <5:
                    segment_imgs.append(img)
                    segment_labs.append(label)
                else:
                    images.append(segment_imgs)
                    labels.append(segment_labs)
                    segment_imgs, segment_labs = [], []
                    segment_imgs.append(img)
                    segment_labs.append(label)
                    fg = 0
                fg += 1
            images.append(segment_imgs)
            labels.append(segment_labs)

        # with open(os.path.join('./data/csv', 'test'+filename), mode='w', newline='') as f:
        #     writer = csv.writer(f)
        #     for i in range(len(images)):
        #         writer.writerow([images[i], labels[i]])
        #     print('writen into csv file:', filename)

        assert len(images) == len(labels)
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):


        imgs, labels = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
        ])
        for i in range(len(imgs)):
            imgs[i] = tf(imgs[i])
            labels[i] = torch.tensor(labels[i])

        imgs_new = torch.rand(imgs[0].size())
        imgs_new = torch.unsqueeze(imgs_new, dim=0).repeat(len(imgs), 1, 1, 1)
        for i in range(len(imgs)):
            imgs_new[i] = imgs[i]

        # label = torch.tensor(labels[0])

        return imgs_new, labels[0]



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


    import visdom
    viz = visdom.Visdom()
    loader_train = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=8)
    for i in range(2):
        print(i)
        for step, (x, y) in enumerate(loader_train):
            viz.images(x[0], win='imgs', nrow=5)
            viz.images(x[1], win='imgs1', nrow=5)
            viz.images(x[2], win='imgs2', nrow=5)
            print(step, x.shape, y.shape)





if __name__ == '__main__':
    main()

