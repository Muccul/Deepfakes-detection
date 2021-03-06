import torch
from dataset import Face
import os
import argparse
from model.xception import net
from model.MesoInception import MesoInception
from torch.utils.data import DataLoader
from tqdm import tqdm

# python test_img.py --data Deepfakes --compression c23 --mode C  --model xception  -c 10
parser =argparse.ArgumentParser(description="Face++ Test")
parser.add_argument("--model", type=str, default="xception", choices=["xception", "mesonet"], help="test model")
parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "All"], help="dataset consist of datas")
parser.add_argument("--batchsize", type=int, default=32, help="Training batch size")
parser.add_argument("--compression", type=str, default="c40", choices=["c23", "c40"])
parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
parser.add_argument("--checkpoint", "-c", type=int, default=0, help="checkpoint of training")
opt = parser.parse_args()


device = torch.device("cuda:0")

def evaluate(model, loader):
    model.eval()

    correct = 0
    total = len(loader.dataset)
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            print('ok:', logits)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


def evaluate_all(model, loader):
    model.eval()

    total_0, total_1, total_2, total_3, total_4 = 0, 0, 0, 0, 0     # Ori, Deepfakes, Face2Face, FaceSwap, NeuralTextures
    correct_0, correct_1, correct_2, correct_3, correct_4 = 0, 0, 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        for n, lab in enumerate(y):
            if y[n] == 0:
                total_0 += 1
                if y[n] == pred[n]:
                    correct_0 += 1
            elif y[n] == 1:
                total_1 += 1
                if y[n] == pred[n]:
                    correct_1 += 1
            elif y[n] == 2:
                total_2 += 1
                if y[n] == pred[n]:
                    correct_2 += 1
            elif y[n] == 3:
                total_3 += 1
                if y[n] == pred[n]:
                    correct_3 += 1
            elif y[n] == 4:
                total_4 += 1
                if y[n] == pred[n]:
                    correct_4 += 1

            if y[n] == 0:
                if pred[n] in [0]:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred[n] in [1, 2, 3, 4]:
                    FN += 1
                else:
                    FP += 1
    return TP, TN, FP, FN, correct_0, correct_1, correct_2, correct_3, correct_4, total_0, total_1, total_2, total_3, total_4


def main():

    weight_path = "weight"
    filename = opt.data + "_" + opt.compression + "_" + opt.mode
    if opt.data == "All":
        if opt.model == "xception":
            model = net(num_class=5)
        else:
            model = MesoInception(out_channel=5)
    else:
        if opt.model == "xception":
            model = net(num_class=2)
        else:
            model = MesoInception(out_channel=2)

    model = torch.nn.DataParallel(model, device_ids=[0]).to(device)
    model.load_state_dict(
        torch.load(os.path.join(weight_path, "model_Mesonet_All_c23_C_10.pth"), map_location=device))
    # if opt.model == "xception":
    #      model.load_state_dict(torch.load(os.path.join(weight_path, "model_%s_%d.pth" % (filename, opt.checkpoint)), map_location=device))
    # else:
    #     model.load_state_dict(torch.load(os.path.join(weight_path, "model_Mesonet_%s_%d.pth" % (filename, opt.checkpoint)), map_location=device))

    dataset_val = Face(mode='val', resize=256 if opt.model=="mesonet" else 299, filename=filename)
    loader_val = DataLoader(dataset_val, batch_size=opt.batchsize, num_workers=8)


    if opt.data == "All":
        TP, TN, FP, FN, correct_0, correct_1, correct_2, correct_3, correct_4, total_0, total_1, total_2, total_3, total_4 = evaluate_all(model, loader_val)
        print("model_%s_%d val:\nOri: %d/%d = %.6f\nDeepfakes: %d/%d = %.6f\nFace2Face: %d/%d = %.6f\nFaceSwap: %d/%d = %.6f\nNeuralTextures: %d/%d = %.6f\nTP:%d, TN:%d, FP:%d, FN:%d\n acc:%.6f" % (
                filename, opt.checkpoint,
                correct_0, total_0, correct_0 / total_0, correct_1, total_1, correct_1 / total_1, correct_2, total_2,
                correct_2 / total_2, correct_3, total_3, correct_3 / total_3, correct_4, total_4, correct_4 / total_4,
                TP, TN, FP, FN, (TP+FN)/(TP+TN+FP+FN)))
        # TP, TN, FP, FN, correct_0, correct_1, correct_2, correct_3, correct_4, total_0, total_1, total_2, total_3, total_4 = evaluate_all(model, loader_test)
        # print("model_%s_%d test:\nOri: %d/%d = %.6f\nDeepfakes: %d/%d = %.6f\nFace2Face: %d/%d = %.6f\nFaceSwap: %d/%d = %.6f\nNeuralTextures: %d/%d = %.6f\nTP:%d, TN:%d, FP:%d, FN:%d\n acc:%.6f" % (
        #         filename, opt.checkpoint,
        #         correct_0, total_0, correct_0 / total_0, correct_1, total_1, correct_1 / total_1, correct_2, total_2,
        #         correct_2 / total_2, correct_3, total_3, correct_3 / total_3, correct_4, total_4, correct_4 / total_4,
        #         TP, TN, FP, FN, (TP + FN) / (TP + TN + FP + FN)))
    else:
        val_acc = evaluate(model, loader_val)
        # val_test = evaluate(model, loader_test)
        print("model_%s_%d：   val_acc：%.6f " % (filename, opt.checkpoint, val_acc))





if __name__ == '__main__':
    main()
