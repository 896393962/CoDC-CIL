"""
#-*-coding:utf-8-*-
# @author: wangyu a beginner programmer, striving to be the strongest.
# @date: 2022/7/7 20:25
"""

def fileName2Txt(image_path, save_path):
    """
    文件夹路径
    img_path = r'F:\dataset\bedroom3w\bedroom3w\bedroomsub'
    txt 保存路径
    save_txt_path = r'F:\dataset\bedroom3w\bedroom_val.txt'
    """

    import os
    # 读取文件夹中的所有文件
    imgs = os.listdir(img_path)

    # 图片名列表
    names = []

    # 过滤：只保留jpg后缀名结尾的图片
    for img in imgs:
        if img.endswith(".jpg"):
            names.append(img)

    print(f'file_sum = {len(names)}')
    txt = open(save_txt_path, 'w')

    for name in names:
        txt.write(name + '\n')  # 逐行写入图片名，'\n'表示换行

    txt.close()

def draw_curve(dirs, i, j):

    import numpy as np
    from matplotlib import pyplot as plt
    data_loss = np.loadtxt(dirs, delimiter=' ', usecols=(0, 2, 4), unpack=True)
    x = data_loss[0]
    y = data_loss[1]
    z = data_loss[2]
    print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    plt.plot(x, y, 'g-', label=u'ans')
    plt.plot(x, z, 'r-', label=u'ans')
    plt.savefig('./loss.jpg')
    plt.show()
    plt.close()

if __name__ == "__main__":
    # 文件夹路径
    img_path = r'F:\dataset\bedroom3w\bedroom3w\bedroomsub'
    # txt 保存路径
    save_txt_path = r'F:\dataset\bedroom3w\bedroom_val.txt'
    dirs = r'F:\dataset\bedroom3w\loss_ans.txt'
    draw_curve(dirs, 0, 1)
    # fileName2Txt(img_path, save_txt_path)
