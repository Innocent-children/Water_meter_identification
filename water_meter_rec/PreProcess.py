import os
import torch
import numpy as np
import cv2
from tqdm import tqdm

# 是否使用 GPU
device = torch.device('cuda')
DEBUG = False


class RecOptions():
    def __init__(self):
        self.height = 32  # 图像尺寸
        self.width = 100
        self.voc_size = 21  # 字符数量 '0123456789ABCDEFGHIJ' + 'PADDING'位
        self.decoder_sdim = 512
        self.max_len = 5  # 文本长度
        self.lr = 1.0
        self.milestones = [40, 60]  # 在第 40 和 60 个 epoch 训练时降低学习率
        self.max_epoch = 80
        self.batch_size = 64
        self.num_workers = 8
        self.print_interval = 25
        self.save_interval = 125
        # self.train_dir = '/MyTest/rec_datasets/train_imgs'
        # self.test_dir = '/MyTest/rec_datasets/test_imgs'
        self.save_dir = '/MyTest/rec_models/'
        self.saved_model_path = '/MyTest/rec_models/checkpoint_final'
        self.rec_res_dir = '/MyTest/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


# rec_args = RecOptions()

# if DEBUG:
#     rec_args.max_epoch = 1
#     rec_args.print_interval = 20
#     rec_args.save_interval = 1
#
#     rec_args.batch_size = 10
#     rec_args.num_workers = 0

'''
标签处理：定义新字符类处理半字符的情况，比如将'0-1半字符'归到'A'类，减小歧义
识别训练数据构造：从完整图像中裁剪出文本图像作为识别模型输入数据
'''


def PreProcess():
    EXT_CHARS = {
        '01': 'A', '12': 'B', '23': 'C', '34': 'D', '45': 'E',
        '56': 'F', '67': 'G', '78': 'H', '89': 'I', '09': 'J'
    }

    train_dir = '/MyTest/train_imgs'
    train_labels_dir = '/MyTest/train_gts'
    word_save_dir = '/MyTest/rec_datasets/train_imgs'  # 保存识别训练数据集
    os.makedirs(word_save_dir, exist_ok=True)
    label_files = os.listdir(train_labels_dir)
    for label_file in tqdm(label_files):
        with open(os.path.join(train_labels_dir, label_file), 'r') as f:
            lines = f.readlines()
        line = lines[0].strip().split()
        locs = line[:8]
        words = line[8:]

        # 标签处理
        if len(words) == 1:  # 区分标签中有一个结果还是两个结果
            ext_word = words[0]
        else:  # 若标签有两个结果，将两个结果相同的部分取出来，与不同的部分组成的数字字符对应的字母拼接起来放入ext_word
            assert len(words) % 2 == 0
            ext_word = ''
            for i in range(len(words[0])):
                char_i = [word[i] for word in words]
                if len(set(char_i)) == 1:
                    ext_word += char_i[0]
                elif len(set(char_i)) == 2:
                    char_i = list(set(char_i))
                    char_i.sort()
                    char_i = ''.join(char_i)
                    ext_char_i = EXT_CHARS[char_i]
                    ext_word += ext_char_i

        locs = [int(t) for t in line[:8]]  # 取出坐标

        # 将倾斜文字图像调整为水平图像
        x1, y1, x2, y2, x3, y3, x4, y4 = locs
        w = int(0.5 * (((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 + (
                (x4 - x3) ** 2 + (y4 - y3) ** 2) ** 0.5))
        h = int(0.5 * (((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5 + (
                (x4 - x1) ** 2 + (y4 - y1) ** 2) ** 0.5))
        src_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype='float32')
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype='float32')
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        image = cv2.imread(os.path.join(train_dir, label_file.replace('.txt', '.jpg')))
        word_image = cv2.warpPerspective(image, M, (w, h))

        # save images
        cv2.imwrite(
            os.path.join(word_save_dir, label_file.replace('.txt', '') + '_' + ext_word + '.jpg'),
            word_image)
