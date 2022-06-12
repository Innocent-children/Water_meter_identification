import os
import torch
import cv2

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
        self.test_dir = '/MyTest/test_imgs'
        self.save_dir = '/MyTest/rec_models/'
        self.saved_model_path = '/MyTest/rec_models/checkpoint_final'
        self.rec_res_dir = '/MyTest/rec_res/'

    def set_(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)


rec_args = RecOptions()
'''
识别结果后处理
'''


def final_postProcess():
    SPECIAL_CHARS = {k: v for k, v in zip('ABCDEFGHIJ', '1234567890')}

    test_dir = '/MyTest/test_imgs'
    rec_res_dir = '/MyTest/rec_res'
    rec_res_files = os.listdir(rec_res_dir)

    final_res = dict()
    for file in os.listdir(test_dir):
        res_file = file.replace('.jpg', '.txt')
        if res_file not in rec_res_files:
            final_res[file] = ''
            continue

        with open(os.path.join(rec_res_dir, res_file), 'r') as f:
            rec_res = f.readline().strip()
        final_res[file] = ''.join(
            [t if t not in 'ABCDEFGHIJ' else SPECIAL_CHARS[t] for t in rec_res])

    with open('/MyTest/rec_res/final_res.txt', 'w') as f:
        for key, value in final_res.items():
            f.write(key + '\t' + value + '\n')


# 生成最终的测试结果
final_postProcess()


def fin():
    '''
    最终结果可视化
    '''
    import matplotlib
    import matplotlib.pyplot as plt

    test_dir = '/MyTest/test_imgs'
    with open('/MyTest/rec_res/final_res.txt', 'r') as f:
        lines = f.readlines()

    plt.figure(figsize=(60, 60))
    lines = lines[:5]
    for i, line in enumerate(lines):
        if len(line.strip().split()) == 1:
            image_name = line.strip()  # 没有识别出来
            word = '###'
        else:
            image_name, word = line.strip().split()
        image = cv2.imread(os.path.join(test_dir, image_name))

        plt.subplot(151 + i)
        plt.title(word, fontdict={'size': 50})
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image)
        plt.show()