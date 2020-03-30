
# coding: utf-8

# In[1]:


import torch
import visdom
import time
import csv, random
import os, glob
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from pretrainedmodels.models.xception import Xception
from PIL import Image
from tqdm import tqdm
from torchsummary import summary


# In[2]:


EPOCH = 3000
BATCH_SIZE = 64
LR = 1e-4

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device_ids = [0, 1]

global_step = 0
best_acc, best_epoch = 0, 0
viz = visdom.Visdom(port=13680)


# # 数据

# In[3]:


class Face(Dataset):
    def __init__(self, roots, resize=299, start=0, end=100, mode='train'):
        super(Face, self).__init__()
        
        self.roots = roots
        self.resize = resize
        self.start = start
        self.end = end
        
        self.name2label = {}
        for step, name in enumerate(roots):
            self.name2label[name.split(os.sep)[-2]] = step
            
        
        self.images, self.labels = self.load_csv('crop_face_deepfakes'+str(start)+'-'+str(end)+'.csv')
        if mode=='train': # 70%
            self.images = self.images[:int(0.7*len(self.images))]
            self.labels = self.labels[:int(0.7*len(self.labels))]
        elif mode=='val': # 15% = 70%->85%
            self.images = self.images[int(0.7*len(self.images)):int(0.85*len(self.images))]
            self.labels = self.labels[int(0.7*len(self.labels)):int(0.85*len(self.labels))]
        elif mode=='test': # 15% = 85%->100%
            self.images = self.images[int(0.85*len(self.images)):]
            self.labels = self.labels[int(0.85*len(self.labels)):]

        
    def load_csv(self, filename):
        print(filename)
        if not os.path.exists(os.path.join('./', filename)):
            images = []
            for face_class in self.roots:
                class_images = []
                for face_imgaes in os.listdir(face_class):
                    class_images += glob.glob(os.path.join(face_class, face_imgaes, '*.png'))
#                 random.shuffle(class_images)
                class_images = class_images[int(self.start/100.0*len(class_images)):int(self.end/100.0*len(class_images))]
                images += class_images
            
            random.shuffle(images)
            with open(os.path.join('./', filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-3]
                    #'./face_crop/original/083/0502.png'.split(os.sep)[-4]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)
                
        images, labels = [], []
        with open(os.path.join('./', filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                #'./face_data/original_sequences/c23/images/083/0078.png', 0
                img ,label = row
                label = int(label)
                
                images.append(img)
                labels.append(label)
                
        assert len(images) == len(labels)
        
        return images, labels
                
        
    def __len__(self):
        
        return len(self.images)
        
    def denormalize(self, x_hat):

        mean = [0.5]*3
        std = [0.5]*3

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x
    
    def __getitem__(self, idx):
        # idx~[0~len(images)]
        # self.images, self.labels
        # './face_data/original_sequences/c23/images/083/0387.png'
        # label: 0
        img, label = self.images[idx], self.labels[idx]
        
        tf = transforms.Compose([
            lambda x:Image.open(x).convert('RGB'), # string path= > image data
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
        img = tf(img)
        label = torch.tensor(label)
        
        return img, label
    


# In[4]:


roots = ['./face_crop/Original/', './face_crop/Deepfakes/']

face_train = Face(roots=roots, start=0, end=30, mode='train')
face_val = Face(roots=roots, start=30, end=60, mode='val')


# In[5]:


train_loader = DataLoader(face_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
val_loader = DataLoader(face_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)


# In[6]:


print(len(train_loader.dataset), next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)


# # Model

# In[7]:


class MesoNet(nn.Module):
    
    def __init__(self):
        super(MesoNet, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=(4, 4))
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 16),
        )
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(16, 2),
        )
        
        
    def Flatten(self, input):
        return input.view(input.size(0), -1)
        
    def forward(self, x):
        x = self.block1(x) 
        x = self.block2(x) 
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.Flatten(x)
        x = self.fc1(x)
        x = self.leakyrelu(x)
        x = self.fc2(x)
        
        return x


# In[8]:


model = MesoNet()


# In[9]:


model = model.to(device)


# # train

# In[10]:


def evaluate(model, loader):
    model.eval()
    
    correct = 0
    total = len(loader.dataset)
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


# In[11]:


criterion =  nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-08)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[1], gamma=0.01)

viz.line([0], [-1], win='Face_crop_loss', opts=dict(title='Face_crop_loss'))
viz.line([0], [-1], win='Face_crop_acc', opts=dict(title='Face_crop_acc'))


# In[12]:


for epoch in range(EPOCH):
    batch_start = time.time()
    epoch_start = time.time()
    for step, (x,y) in enumerate(train_loader):
        model.train()
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        viz.line([loss.item()], [global_step], win='Face_crop_loss', update='append')
        global_step += 1
        
        if global_step % 10 == 0:
            batch_end = time.time()
            print('Epoch{0:2d}-{1:2d}% :time:{2}'.format(epoch, int(step*BATCH_SIZE/len(train_loader.dataset)*100), batch_end-batch_start))
            batch_start = time.time()
        
    
    if epoch % 1 == 0:
        epoch_end = time.time()
        print('epoch {0:3d} time:{1}'.format(epoch,epoch_end-epoch_start))
        acc = evaluate(model, val_loader)
        viz.line([acc], [epoch], win='Face_crop_acc', update='append')
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(),'../Model/Crop30%DeepFakes-MesoNet2.pth')

