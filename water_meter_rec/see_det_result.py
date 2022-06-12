import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def see():
    # # 检测结果可视化
    test_dir = '/MyTest/test_imgs'
    det_dir = '/MyTest/det_res'
    det_vis_dir = '/MyTest/det_vis_test'

    os.makedirs(det_vis_dir, exist_ok=True)
    label_files = os.listdir(det_dir)
    cnt = 0
    plt.figure(figsize=(60, 60))
    for label_file in tqdm(label_files):
        if not label_file.endswith('.txt'):
            continue
        image = cv2.imread(os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))

        with open(os.path.join(det_dir, label_file), 'r') as f:
            lines = f.readlines()

        save_name = label_file.replace('det_res_', '')[:-4] + '.jpg'
        if len(lines) == 0:
            cv2.imwrite(os.path.join(det_vis_dir, save_name), image)
        else:
            line = lines[0].strip().split(',')
            locs = [float(t) for t in line[:8]]

            # draw box
            locs = np.array(locs).reshape(1, -1, 2).astype(np.int32)
            image = cv2.imread(
                os.path.join(test_dir, label_file.replace('det_res_', '')[:-4] + '.jpg'))
            cv2.polylines(image, locs, True, (255, 255, 0), 8)  # OpenCV使用BGR格式，8代表线条的粗细

            # save images
            save_name = label_file.replace('det_res_', '')[:-4] + '.jpg'
            cv2.imwrite(os.path.join(det_vis_dir, save_name), image)

        if cnt < 5:  # 只画5张
            plt.subplot(151 + cnt)
            plt.title(save_name, fontdict={'size': 60})
            plt.xticks([])
            plt.yticks([])
            plt.imshow(image)
            plt.show()
            cnt += 1
