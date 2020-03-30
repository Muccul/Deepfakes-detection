import cv2
import os
import argparse
from tqdm import tqdm

#python video2img.py --data Original --compression c40 --mode F
parser = argparse.ArgumentParser(description="video2img")
parser.add_argument("--frame_num", type=int, default=20, help="frame number")
parser.add_argument("--frame_val", type=int, default=6, help="frame interval number")
parser.add_argument("--data", type=str, default="Deepfakes", choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "Original"])
parser.add_argument("--compression", type=str, default="c23", choices=["c23", "c40"])
parser.add_argument("--mode", type=str, default="C", choices=["F", "C"], help="F：video2full_img;  C：video2crop_img")
opt = parser.parse_args()


def get_box_edge(full_img):
    """
    获取mask边框
    return  左上点(x1,y1)、右下点(x2,y2)
    """
    h = full_img.shape[0]
    w = full_img.shape[1]
    x1, y1, x2, y2 = 0, 0, 0, 0
    first_flag = 0
    num = 0
    for x in range(h):
        for y in range(w):
            if not full_img[x][y][0] == 0:
                x1 = x
                y1 = y
                first_flag = 1
                break
        if first_flag:
            break
    # 确定y2
    w_flag = 0
    for y in range(y1, w):
        if full_img[x1][y][0] == 0:
            y2 = y
            w_flag = 1
            break
    if w_flag == 0:
        y2 = w
    # 确定x2
    h_flag = 0
    for x in range(x1, h):
        if full_img[x][y1][0] == 0:
            x2 = x
            h_flag = 1
            break
    if h_flag == 0:
        x2 = h

    return x1, y1, x2, y2


def get_boundingbox(x1, y1, x2, y2, mask_frame, scale=1.4, minsize=None):
    """
    以中心扩大mask掩码的scale倍
    return: x, y, bounding_box_size in opencv form
    """
    height = mask_frame.shape[0]
    width = mask_frame.shape[1]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb


def output_face(src_frame, mask_frame, filename, resize=299, scale=1.3):
    """
    输入sec、mask帧图片，输出裁剪、resize的人脸图片
    """
    x1, y1, x2, y2 = get_box_edge(mask_frame)
    if x1 == 0 and x2 == 0:
        return False
    x3, y3, size3 = get_boundingbox(y1, x1, y2, x2, mask_frame)
    tmp_img = src_frame[y3:y3 + size3, x3:x3 + size3]
    tmp_img = cv2.resize(tmp_img, (resize, resize))
    cv2.imwrite(filename, tmp_img)
    return True


def BGR2RGR(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def video2full_img():
    video_root = os.path.join("data", opt.data, opt.compression+"/videos")
    out_root = os.path.join("data", opt.data, opt.compression+"/full_images")
    os.makedirs(out_root, exist_ok=True)

    video_files = os.listdir(video_root)
    for f in tqdm(video_files):
        file_name = f.split(".")[0]
        file_root = os.path.join(out_root, file_name)
        os.makedirs(file_root, exist_ok=True)

        video_reader = cv2.VideoCapture(os.path.join(video_root, f))
        num = 1
        save_num = 1
        while video_reader.isOpened():
            success, fram = video_reader.read()
            if not success or save_num > opt.frame_num:
                break
            if num % opt.frame_val == 0:
                cv2.imwrite(os.path.join(file_root, '{:04d}.png'.format(save_num)), fram)
                save_num += 1
            num += 1

def video2crop_img():
    video_root = os.path.join("data", opt.data, opt.compression+"/videos")
    mask_root = os.path.join("data", "manipulated_sequences/Deepfakes", "masks/videos")
    out_root = os.path.join("data", opt.data, opt.compression+"/crop_images")
    os.makedirs(out_root, exist_ok=True)

    video_files = os.listdir(video_root)
    mask_files  = os.listdir(mask_root)
    video_files.sort()
    mask_files.sort()

    for f, f_mask in tqdm(zip(video_files, mask_files)):
        file_name = f.split(".")[0]
        file_root = os.path.join(out_root, file_name)

        os.makedirs(file_root, exist_ok=True)

        video_reader = cv2.VideoCapture(os.path.join(video_root, f))
        mask_reader = cv2.VideoCapture(os.path.join(mask_root, f_mask))
        num = 1
        save_num = 1
        while video_reader.isOpened():
            success, video_frame = video_reader.read()
            _, mask_frame = mask_reader.read()
            if not success or save_num > opt.frame_num:
                break
            if num % opt.frame_val == 0:
                output_face(video_frame, mask_frame, os.path.join(file_root, '{:04d}.png'.format(save_num)))
                save_num += 1
            num += 1

def main():

    if opt.data == "Original":
        opt.data = os.path.join("original_sequences")
    else:
        opt.data = os.path.join("manipulated_sequences", opt.data)

    if opt.mode == "F":
        video2full_img()
    elif opt.mode == "C":
        video2crop_img()



if __name__ == '__main__':
    main()