import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import visdom
from dataset_tsm import Face
from model.tsm import TSM

#  python train_tsm.py --modality rgbdiff --n_segment 4 --data NeuralTextures --compression c40 --mode C --checkpoint 0
parser =argparse.ArgumentParser(description="DFDC_TSM Train")
parser.add_argument("--modality", type=str, default='rgb', choices=['rgb', 'rgbdiff'], help='modality')
parser.add_argument("--n_segment", type=int, default=5, help="number of segment")
parser.add_argument("--batchsize", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int,  default=12, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=5e-4, help="learning rate")
parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "All"], help="dataset consist of datas")
parser.add_argument("--compression", type=str, default="c40", choices=["c23", "c40"])
parser.add_argument("--checkpoint", "-c", type=int, default=0, help="checkpoint of training")
parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
opt = parser.parse_args()


device = torch.device("cuda:0")


def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def main():

    if opt.data == "All":
        model = TSM(num_classes=5, n_segment=opt.n_segment)
    else:
        model = TSM(num_classes=2, n_segment=opt.n_segment)

    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    for m in model.modules():
        if isinstance(m, (nn.Linear)):
            print("init Linear")
            nn.init.orthogonal_(m.weight)
            # nn.init.kaiming_normal_(m.weight, mode='fan_in')
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 40], gamma=0.1)
    criterion = nn.CrossEntropyLoss().to(device)

    filename = opt.data + "_" + opt.compression + "_" + opt.mode
    dataset_train = Face(mode='train', filename=filename, resize=224, modality=opt.modality)
    dataset_val = Face(mode='val', filename=filename, resize=224, modality=opt.modality)
    loader_train = DataLoader(dataset_train, batch_size=opt.batchsize, shuffle=True, num_workers=8, drop_last=True)
    loader_val = DataLoader(dataset_val, batch_size=opt.batchsize, num_workers=8)

    dataset_step = len(loader_train.dataset)/(opt.batchsize)


    viz = visdom.Visdom()
    weight_path = "weight"
    if opt.checkpoint != 0:
        model.load_state_dict(torch.load(os.path.join(weight_path, "tsm_%s_%s_%d.pth" %(opt.modality,filename, opt.checkpoint))))
        viz.line([0.2], [dataset_step * opt.checkpoint], win='loss', opts=dict(title='loss'))
        viz.line([0.9], [opt.checkpoint], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0.9], [opt.checkpoint], win='train_acc', opts=dict(title='train_acc'))
    print('check point:%d' %(opt.checkpoint))

    if opt.checkpoint == 0:
        viz.line([0.2], [dataset_step * opt.checkpoint], win='loss', opts=dict(title='loss'))
        viz.line([0.9], [opt.checkpoint], win='val_acc', opts=dict(title='val_acc'))
        viz.line([0.9], [opt.checkpoint], win='train_acc', opts=dict(title='train_acc'))
    global_step = dataset_step*opt.checkpoint

    for epoch in range(opt.checkpoint, opt.epochs):
        model.train()
        train_acc_all = 0
        if epoch < 5:
            current_lr = opt.lr
        if epoch >= 5:
            current_lr = opt.lr/10

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        for step, (x,y) in enumerate(loader_train):
            x, y = x.to(device), y.to(device)

            x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
            y_hat = model(x)

            loss_cls = criterion(y_hat, y)

            # loss_coherence = 0.2
            #loss_coherence = torch.abs(y_div - y_hat.unsqueeze(dim=1)).mean()
            # print(y_div[0], y_hat.unsqueeze(dim=1)[0])
            # loss_coherence = sum([torch.abs(y_div[:,i] - y_div[:,i+1]).mean() for i in range(y_div.size(1)-1)])
            loss = 1.0*loss_cls # + 0.1*loss_coherence

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
            torch.save(model.state_dict(), os.path.join(weight_path, "tsm_%s_%s_%d.pth" %(opt.modality, filename, epoch+1)))
            with open("log/tsm_%s_%s.txt" %(opt.modality, filename), "a+") as f:
                f.write("epoch%d:   train_acc：%.6f,   val_acc：%.6f \n" % (epoch+1, train_acc_all/((step+1)*opt.batchsize), val_acc))


if __name__ == '__main__':
    main()
