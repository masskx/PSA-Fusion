import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from natsort import natsorted
from tqdm import tqdm


from utils.eval_one_method import write_excel, evaluation_one


def save_images_from_tensors(ir_batch, vi_batch, f_batch, epoch_number):
    # 检查是否有必要创建文件夹
    batch_folder = f"epoch-{epoch_number}"
    if not os.path.exists("../exp/result/" + batch_folder):
        # print("保存图片的文件夹不存在创建一个")
        os.makedirs("../exp/result/" + batch_folder)
        os.makedirs("../exp/result/" + batch_folder + "/vi/")
        os.makedirs("../exp/result/" + batch_folder + "/ir/")
        os.makedirs("../exp/result/" + batch_folder + "/f/")
        os.makedirs("../exp/result/" + batch_folder + "/Metric/")
    # 提取第一张图片
    # 获取当前批次的数量
    batch_size = f_batch.shape[0]
    # print(f"batch大小{batch_size}")
    # 遍历整个批次

    for i in range(batch_size):
        # 提取每张图像
        ir_image = ir_batch[i].cpu().detach().numpy().squeeze()
        vi_image = vi_batch[i].cpu().detach().numpy().squeeze()
        f_image = f_batch[i].cpu().detach().numpy().squeeze()
        # 设置保存路径
        v_path = os.path.join("../exp/result/" + batch_folder, f"vi/{i}.png")
        i_path = os.path.join("../exp/result/" + batch_folder, f"ir/{i}.png")
        f_path = os.path.join("../exp/result/" + batch_folder, f"f/{i}.png")
        # 评测指标
        plt.imsave(fname=i_path, arr=ir_image, cmap='gray')
        plt.imsave(fname=v_path, arr=vi_image, cmap='gray')
        plt.imsave(fname=f_path, arr=f_image, cmap='gray')
    #########################################################################################################
    with_mean = True
    EN_list = []
    MI_list = []
    SF_list = []
    AG_list = []
    SD_list = []
    CC_list = []
    SCD_list = []
    VIF_list = []
    MSE_list = []
    PSNR_list = []
    Qabf_list = []
    Nabf_list = []
    SSIM_list = []
    MS_SSIM_list = []
    filename_list = ['']
    ir_dir = os.path.join("../exp/result/" + batch_folder + "/ir/")
    vi_dir = os.path.join("../exp/result/" + batch_folder + "/vi/")
    f_dir = os.path.join("../exp/result/" + batch_folder + "/f/")
    Method = 'AFGAN'
    save_dir = os.path.join("../exp/result/" + batch_folder + "/Metric/")
    os.makedirs(save_dir, exist_ok=True)
    metric_save_name = os.path.join(save_dir, 'metric_{}.xlsx'.format(epoch_number))
    filelist = natsorted(os.listdir(f_dir))
    eval_bar = tqdm(filelist)
    for _, item in enumerate(eval_bar):
        ir_name = os.path.join(ir_dir, item)
        vi_name = os.path.join(vi_dir, item)
        f_name = os.path.join(f_dir, item)
        EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(ir_name, vi_name,
                                                                                                f_name)
        EN_list.append(EN)
        MI_list.append(MI)
        SF_list.append(SF)
        AG_list.append(AG)
        SD_list.append(SD)
        CC_list.append(CC)
        SCD_list.append(SCD)
        VIF_list.append(VIF)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        Qabf_list.append(Qabf)
        Nabf_list.append(Nabf)
        SSIM_list.append(SSIM)
        MS_SSIM_list.append(MS_SSIM)
        filename_list.append(item)
        eval_bar.set_description("{} | {}".format(Method, item))
    if with_mean:
        # 添加均值
        EN_list.append(np.mean(EN_list))
        MI_list.append(np.mean(MI_list))
        SF_list.append(np.mean(SF_list))
        AG_list.append(np.mean(AG_list))
        SD_list.append(np.mean(SD_list))
        CC_list.append(np.mean(CC_list))
        SCD_list.append(np.mean(SCD_list))
        VIF_list.append(np.mean(VIF_list))
        MSE_list.append(np.mean(MSE_list))
        PSNR_list.append(np.mean(PSNR_list))
        Qabf_list.append(np.mean(Qabf_list))
        Nabf_list.append(np.mean(Nabf_list))
        SSIM_list.append(np.mean(SSIM_list))
        MS_SSIM_list.append(np.mean(MS_SSIM_list))
        filename_list.append('mean')

        ## 添加标准差
        EN_list.append(np.std(EN_list))
        MI_list.append(np.std(MI_list))
        SF_list.append(np.std(SF_list))
        AG_list.append(np.std(AG_list))
        SD_list.append(np.std(SD_list))
        CC_list.append(np.std(CC_list[:-1]))
        SCD_list.append(np.std(SCD_list))
        VIF_list.append(np.std(VIF_list))
        MSE_list.append(np.std(MSE_list))
        PSNR_list.append(np.std(PSNR_list))
        Qabf_list.append(np.std(Qabf_list))
        Nabf_list.append(np.std(Nabf_list))
        SSIM_list.append(np.std(SSIM_list))
        MS_SSIM_list.append(np.std(MS_SSIM_list))
        filename_list.append('std')

    ## 保留三位小数
    EN_list = [round(x, 3) for x in EN_list]
    MI_list = [round(x, 3) for x in MI_list]
    SF_list = [round(x, 3) for x in SF_list]
    AG_list = [round(x, 3) for x in AG_list]
    SD_list = [round(x, 3) for x in SD_list]
    CC_list = [round(x, 3) for x in CC_list]
    SCD_list = [round(x, 3) for x in SCD_list]
    VIF_list = [round(x, 3) for x in VIF_list]
    MSE_list = [round(x, 3) for x in MSE_list]
    PSNR_list = [round(x, 3) for x in PSNR_list]
    Qabf_list = [round(x, 3) for x in Qabf_list]
    Nabf_list = [round(x, 3) for x in Nabf_list]
    SSIM_list = [round(x, 3) for x in SSIM_list]
    MS_SSIM_list = [round(x, 3) for x in MS_SSIM_list]

    EN_list.insert(0, '{}'.format(Method))
    MI_list.insert(0, '{}'.format(Method))
    SF_list.insert(0, '{}'.format(Method))
    AG_list.insert(0, '{}'.format(Method))
    SD_list.insert(0, '{}'.format(Method))
    CC_list.insert(0, '{}'.format(Method))
    SCD_list.insert(0, '{}'.format(Method))
    VIF_list.insert(0, '{}'.format(Method))
    MSE_list.insert(0, '{}'.format(Method))
    PSNR_list.insert(0, '{}'.format(Method))
    Qabf_list.insert(0, '{}'.format(Method))
    Nabf_list.insert(0, '{}'.format(Method))
    SSIM_list.insert(0, '{}'.format(Method))
    MS_SSIM_list.insert(0, '{}'.format(Method))
    write_excel(metric_save_name, 'EN', 0, filename_list)
    write_excel(metric_save_name, "MI", 0, filename_list)
    write_excel(metric_save_name, "SF", 0, filename_list)
    write_excel(metric_save_name, "AG", 0, filename_list)
    write_excel(metric_save_name, "SD", 0, filename_list)
    write_excel(metric_save_name, "CC", 0, filename_list)
    write_excel(metric_save_name, "SCD", 0, filename_list)
    write_excel(metric_save_name, "VIF", 0, filename_list)
    write_excel(metric_save_name, "MSE", 0, filename_list)
    write_excel(metric_save_name, "PSNR", 0, filename_list)
    write_excel(metric_save_name, "Qabf", 0, filename_list)
    write_excel(metric_save_name, "Nabf", 0, filename_list)
    write_excel(metric_save_name, "SSIM", 0, filename_list)
    write_excel(metric_save_name, "MS_SSIM", 0, filename_list)
    write_excel(metric_save_name, 'EN', 1, EN_list)
    write_excel(metric_save_name, 'MI', 1, MI_list)
    write_excel(metric_save_name, 'SF', 1, SF_list)
    write_excel(metric_save_name, 'AG', 1, AG_list)
    write_excel(metric_save_name, 'SD', 1, SD_list)
    write_excel(metric_save_name, 'CC', 1, CC_list)
    write_excel(metric_save_name, 'SCD', 1, SCD_list)
    write_excel(metric_save_name, 'VIF', 1, VIF_list)
    write_excel(metric_save_name, 'MSE', 1, MSE_list)
    write_excel(metric_save_name, 'PSNR', 1, PSNR_list)
    write_excel(metric_save_name, 'Qabf', 1, Qabf_list)
    write_excel(metric_save_name, 'Nabf', 1, Nabf_list)
    write_excel(metric_save_name, 'SSIM', 1, SSIM_list)
    write_excel(metric_save_name, 'MS_SSIM', 1, MS_SSIM_list)

    # # 先存再计算
    # EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM = evaluation_one(i_path, v_path, f_path)
    # return EN, MI, SF, AG, SD, CC, SCD, VIF, MSE, PSNR, Qabf, Nabf, SSIM, MS_SSIM
