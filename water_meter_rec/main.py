import torch

import PreProcess
import det_train
import rec_test
import rec_train
import see_finally_result

device = torch.device('cuda')

DEBUG = False  # Debug模式可快速跑通代码，非Debug模式可得到更好的结果
# 检测和识别模型需要足够的训练迭代次数，因此DEBUG模式下几乎得不到最终有效结果


# 运行训练代码  文字检测模型训练     detect
# det_train.det_train()                       #产生det_models文件夹
# 运行测试代码     文字检测模型测试    detect
# det_test.det_test()             #产生det_res文件夹
# 检测结果可视化
# see_det_result.see()              #产生det_vis_test文件夹
# 运行识别训练数据前处理代码
# PreProcess.PreProcess()  # 产生rec_datasets下的train_imgs文件夹
# 运行训练代码    文字识别模型训练      recognize
# rec_train.rec_train()                  #产生rec_models文件夹
# 运行测试代码       文字识别模型测试  recognize
# rec_test.rec_test_data_gen()  # 产生rec_datasets下的test_imgs文件夹
# rec_test.rec_test()  # 产生rec_res文件夹
# 生成最终的测试结果
# see_finally_result.final_postProcess()
see_finally_result.fin()
