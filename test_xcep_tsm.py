import torch
from dataset_tsm import Face
import os
import argparse
from model.tsm import TSM
from model.xception import net
from torch.utils.data import DataLoader
from tqdm import tqdm

# python test_xcep_tsm.py --data NeuralTextures --compression c40 --mode C -c_rgb 8 -c_diff 7 -c_xcep 10
parser =argparse.ArgumentParser(description="Face++ TSM Test")
parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "All"], help="dataset consist of datas")
parser.add_argument("--batchsize", type=int, default=16, help="Training batch size")
parser.add_argument("--compression", type=str, default="c40", choices=["c23", "c40"])
parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
parser.add_argument("--checkpoint_rgb", "-c_rgb", type=int, default=0, help="checkpoint of training")
parser.add_argument("--checkpoint_diff", "-c_diff", type=int, default=0, help="checkpoint of training")
parser.add_argument("--checkpoint_xcep", "-c_xcep", type=int, default=0, help="checkpoint of training")
opt = parser.parse_args()


device = torch.device("cuda:0")

def evaluate(model_rgb, model_diff, model_xcep, loader_rgb, loader_diff, loader_xcep):
    model_rgb.eval()
    model_diff.eval()
    model_xcep.eval()

    correct_rgb = 0
    correct_diff = 0
    correct_xcep = 0
    correct_avg = 0
    total = len(loader_rgb.dataset)
    for (x_rgb, y_rgb), (x_diff, y_diff), (x_xcep, y_xcep) in tqdm(zip(loader_rgb, loader_diff, loader_xcep)):
        x_rgb, x_diff, x_xcep, y_xcep, y_rgb,  = x_rgb.to(device), x_diff.to(device), x_xcep.to(device), \
                                                 y_xcep.to(device), y_rgb.to(device)
        x_rgb = x_rgb.view(-1, x_rgb.size(-3), x_rgb.size(-2), x_rgb.size(-1))
        x_diff = x_diff.view(-1, x_diff.size(-3), x_diff.size(-2), x_diff.size(-1))
        b, t = x_xcep.size(0), x_xcep.size(1)
        x_xcep = x_xcep.view(b * t, x_xcep.size(-3), x_xcep.size(-2), x_xcep.size(-1))
        with torch.no_grad():
            logits_rgb = model_rgb(x_rgb)
            logits_diff = model_diff(x_diff)
            logits_xcep = model_xcep(x_xcep).view(b, t, -1)
            logits_xcep = logits_xcep.mean(dim=1)
            logits_avg = (logits_rgb + logits_diff + logits_xcep) / 3

            pred_xcep = logits_xcep.argmax(dim=1)
            pred_rgb = logits_rgb.argmax(dim=1)
            pred_diff = logits_diff.argmax(dim=1)
            pred_avg = logits_avg.argmax(dim=1)
        correct_rgb += torch.eq(pred_rgb, y_rgb).sum().float().item()
        correct_diff += torch.eq(pred_diff, y_rgb).sum().float().item()
        correct_xcep += torch.eq(pred_xcep, y_rgb).sum().float().item()
        correct_avg += torch.eq(pred_avg, y_rgb).sum().float().item()
    return correct_rgb / total, correct_diff / total, correct_xcep / total, correct_avg / total


def evaluate_all(model_rgb, model_diff, loader_rgb, loader_diff):
    model_rgb.eval()
    model_diff.eval()


    total_0, total_1, total_2, total_3, total_4 = 0, 0, 0, 0, 0     # Ori, Deepfakes, Face2Face, FaceSwap, NeuralTextures
    correct_0, correct_1, correct_2, correct_3, correct_4 = 0, 0, 0, 0, 0
    TP, TN, FP, FN = 0, 0, 0, 0
    for (x_rgb, y_rgb), (x_diff, y_diff) in tqdm(zip(loader_rgb, loader_diff)):
        x_rgb, x_diff, y_rgb = x_rgb.to(device), x_diff.to(device), y_rgb.to(device)
        x_rgb = x_rgb.view(-1, x_rgb.size(-3), x_rgb.size(-2), x_rgb.size(-1))
        x_diff = x_diff.view(-1, x_diff.size(-3), x_diff.size(-2), x_diff.size(-1))
        with torch.no_grad():
            logits_rgb = model_rgb(x_rgb)
            logits_diff = model_diff(x_diff)
            logits_avg = (logits_rgb + logits_diff) / 2
            # pred_rgb = logits_rgb.argmax(dim=1)
            # pred_diff = logits_diff.argmax(dim=1)
            pred_avg = logits_avg.argmax(dim=1)
        for n, lab in enumerate(y_rgb):
            if y_rgb[n] == 0:
                total_0 += 1
                if y_rgb[n] == pred_avg[n]:
                    correct_0 += 1
            elif y_rgb[n] == 1:
                total_1 += 1
                if y_rgb[n] == pred_avg[n]:
                    correct_1 += 1
            elif y_rgb[n] == 2:
                total_2 += 1
                if y_rgb[n] == pred_avg[n]:
                    correct_2 += 1
            elif y_rgb[n] == 3:
                total_3 += 1
                if y_rgb[n] == pred_avg[n]:
                    correct_3 += 1
            elif y_rgb[n] == 4:
                total_4 += 1
                if y_rgb[n] == pred_avg[n]:
                    correct_4 += 1

            if y_rgb[n] == 0:
                if pred_avg[n] in [0]:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred_avg[n] in [1, 2, 3, 4]:
                    FN += 1
                else:
                    FP += 1
    return TP, TN, FP, FN, correct_0, correct_1, correct_2, correct_3, correct_4, total_0, total_1, total_2, total_3, total_4


def main():

    weight_path = "weight"
    filename = opt.data + "_" + opt.compression + "_" + opt.mode
    if opt.data == "All":
        model_rgb = TSM(num_classes=5, n_segment=5)
        model_diff = TSM(num_classes=5, n_segment=4)
        model_xcep = net(num_class=5)
    else:
        model_rgb = TSM(num_classes=2, n_segment=5)
        model_diff = TSM(num_classes=2, n_segment=4)
        model_xcep = net(num_class=2)


    model_rgb = torch.nn.DataParallel(model_rgb, device_ids=[0]).to(device)
    model_diff = torch.nn.DataParallel(model_diff, device_ids=[0]).to(device)
    model_xcep = torch.nn.DataParallel(model_xcep, device_ids=[0]).to(device)

    model_rgb.load_state_dict(
        torch.load(os.path.join(weight_path, "tsm_%s_%s_%d.pth" %('rgb', filename, opt.checkpoint_rgb)), map_location=device))
    model_diff.load_state_dict(
        torch.load(os.path.join(weight_path, "tsm_%s_%s_%d.pth" % ('rgbdiff', filename, opt.checkpoint_diff)), map_location=device))
    model_xcep.load_state_dict(
        torch.load(os.path.join(weight_path, "model_%s_%d.pth" % (filename, opt.checkpoint_xcep)), map_location=device))


    dataset_val_rgb = Face(mode='val', resize=224, filename=filename, modality='rgb')
    dataset_val_diff = Face(mode='val', resize=224, filename=filename, modality='rgbdiff')
    dataset_val_xcep = Face(mode='val', resize=299, filename=filename)
    loader_val_rgb = DataLoader(dataset_val_rgb, batch_size=opt.batchsize, num_workers=8)
    loader_val_diff = DataLoader(dataset_val_diff, batch_size=opt.batchsize, num_workers=8)
    loader_val_xcep = DataLoader(dataset_val_xcep, batch_size=opt.batchsize, num_workers=8)

    if opt.data == "All":
        pass
        TP, TN, FP, FN, correct_0, correct_1, correct_2, correct_3, correct_4, total_0, total_1, total_2, total_3, total_4 = evaluate_all(model_rgb, model_diff, loader_val_rgb, loader_val_diff)
        print("model_%s_%d val:\nOri: %d/%d = %.6f\nDeepfakes: %d/%d = %.6f\nFace2Face: %d/%d = %.6f\nFaceSwap: %d/%d = %.6f\nNeuralTextures: %d/%d = %.6f\nTP:%d, TN:%d, FP:%d, FN:%d\n acc:%.6f" % (
                filename, opt.checkpoint_rgb,
                correct_0, total_0, correct_0 / total_0, correct_1, total_1, correct_1 / total_1, correct_2, total_2,
                correct_2 / total_2, correct_3, total_3, correct_3 / total_3, correct_4, total_4, correct_4 / total_4,
                TP, TN, FP, FN, (TP+FN)/(TP+TN+FP+FN)))

    else:
        rgb_acc, diff_acc, xcep_acc, avg_acc = evaluate(model_rgb, model_diff, model_xcep, loader_val_rgb, loader_val_diff, loader_val_xcep)
        print("model_%s_%d：   rgb_acc：%.6f  diff_acc：%.6f  xcep_acc：%.6f avg_acc：%.6f" % (filename, opt.checkpoint_rgb, rgb_acc, diff_acc, xcep_acc, avg_acc))




if __name__ == '__main__':
    main()
