import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import visdom
from model.MesoInception import MesoInception
from dataset import Face

#  python train_mesonet.py --data Deepfakes --compression c23 --mode C --checkpoint 0
parser =argparse.ArgumentParser(description="DFDC Train")
parser.add_argument("--batchsize", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int,  default=10, help="Number of training epochs")
parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "All"], help="dataset consist of datas")
parser.add_argument("--compression", type=str, default="c40", choices=["c23", "c40"])
parser.add_argument("--checkpoint", "-c", type=int, default=0, help="checkpoint of training")
parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
opt = parser.parse_args()


device = torch.device("cuda:3")

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():

    if opt.data == "All":
        model = MesoInception(out_channel=5)
    else:
        model = MesoInception(out_channel=2)
    model = torch.nn.DataParallel(model, device_ids=[3]).to(device)
    optimier = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimier, milestones=[5, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    filename = opt.data + "_" + opt.compression + "_" + opt.mode
    dataset_train = Face(mode='train', filename=filename, resize=256)
    dataset_val = Face(mode='val', filename=filename, resize=256)
    loader_train = DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=True)

    dataset_step = len(loader_train.dataset)/opt.batchsize

    viz = visdom.Visdom(port=13680)
    weight_path = "weight"
    if opt.checkpoint != 0:
        model.load_state_dict(torch.load(os.path.join(weight_path, "model_Mesonet_%s_%d.pth" %(filename, opt.checkpoint))))
        viz.line([0.2], [dataset_step * opt.checkpoint], win='loss', opts=dict(title='loss'))
        viz.line([0.9], [opt.checkpoint], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0.9], [opt.checkpoint], win='train_acc', opts=dict(title='train_acc'))
    print('check point:%d' %(opt.checkpoint))

    if opt.checkpoint == 0:
        viz.line([0.2], [dataset_step * opt.checkpoint], win='loss', opts=dict(title='loss'))
        viz.line([0.9], [opt.checkpoint], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0.9], [opt.checkpoint], win='train_acc', opts=dict(title='train_acc'))
    global_step = dataset_step*opt.checkpoint

    model.train()
    for epoch in range(opt.checkpoint, opt.epochs):
        train_acc_all = 0
        for step, (x,y) in enumerate(loader_train):
            x, y = x.to(device), y.to(device)
            y_hat = model(x)

            loss = criterion(y_hat, y)

            optimier.zero_grad()
            loss.backward()
            optimier.step()

            pred = y_hat.argmax(dim=1)
            train_acc_all += torch.eq(pred, y).sum().float().item()

            if step % 10 == 0 and step != 0:
                viz.line([loss.item()], [global_step], win='loss', update='append')
                print("[epoch %d][%d/%d]  \n all_loss: %.6f \n" %(epoch + 1, step, dataset_step, loss.item()))
            global_step +=1

        if epoch % 1 == 0:
            print("****************************\nepoch:{} train_acc:{}\n***************************************\n".format(epoch + 1, train_acc_all/((step+1)*opt.batchsize)))
            val_acc = evaluate(model, loader_val)
            print("****************************\nepoch:{} val_acc:{}\n***************************************\n".format(epoch + 1, val_acc))
            viz.line([val_acc], [epoch+1], win='val_acc', update='append')
            viz.line([train_acc_all/((step+1)*opt.batchsize)], [epoch+1], win='train_acc', update='append')
            torch.save(model.state_dict(), os.path.join(weight_path, "model_Mesonet_%s_%d.pth" %(filename, epoch+1)))
            with open("log/Mesonet_%s.txt" %(filename), "a+") as f:
                f.write("epoch%d:   train_acc：%.6f,   val_acc：%.6f \n" % (epoch+1, train_acc_all/((step+1)*opt.batchsize), val_acc))
            scheduler.step()


if __name__ == '__main__':
    main()
