import os
import numpy as np
from PIL import Image
import h5py
import torch
import natsort
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


'''
我的数据形式
dataset
    OCTA_3M
        OCT
            10301
                1.bmp
                2.bmp
                3.bmp
                ...
        OCTA
            10301
                1.bmp
                2.bmp
                3.bmp
                ...
    OCTA_6M
        OCT
            10001
                1.bmp
                2.bmp
                3.bmp
                ...
        OCTA
            10001
                1.bmp
                2.bmp
                3.bmp
                ...
'''


def 读数据集图像路径(数据集根目录):
    """
    传入数据集根目录，返回保存图像路径的字典。
    6M数据，图像目录范围10001-10300，共300组OCTA体积数据，形状400*400*640
    3M数据，图像目录范围10301-10500，共200组OCTA体积数据，形状304*304*640

    :param 数据集根目录: 数据集根目录下包含两个文件夹：OCTA_3M和OCTA_6M
    :return:返回两个保存图像路径的字典。训练集['OCTA_3M'], 训练集['OCTA_6M']
    """
    # 因为所有数据用来自监督训练所以没有训练和验证的区别
    训练集 = {}  # 空字典用来保存图像路径，暂时只考虑6M数据，但实际两种都读取了
    for 不同大小OCTA in os.listdir(数据集根目录):
        训练集.update({不同大小OCTA: {}})  # 将OCTA_3M和OCTA_6M分别以空字典形式加入数据集字典中得到 训练集 = {'OCTA_3M':{}, 'OCTA_6M':{}}
        for 图像模态 in os.listdir(os.path.join(数据集根目录, 不同大小OCTA)):
            # 将OCT和OCTA分别以空字典形式加入，得到{'OCTA_3M':{'OCT':{}, 'OCTA':{}}, 'OCTA_6M':{'OCT':{}, 'OCTA':{}}}
            训练集[不同大小OCTA].update({图像模态: {}})
            立体数据文件夹列表 = os.listdir(os.path.join(数据集根目录, 不同大小OCTA, 图像模态))
            立体数据文件夹列表 = natsort.natsorted(立体数据文件夹列表)  # 将3维图像数据的文件夹名称按照自然数顺序排序 如10301~10500
            for 立体数据名称 in 立体数据文件夹列表:
                训练集[不同大小OCTA][图像模态].update({立体数据名称: {}})  # 将每组立体数据名称加入字典
                B_scan列表 = os.listdir(os.path.join(数据集根目录, 不同大小OCTA, 图像模态, 立体数据名称))
                B_scan列表 = natsort.natsorted(B_scan列表)  # 将每张B-scan图像名称按照自然数顺序排序
                for i in range(0, len(B_scan列表)):
                    # 拼接得到每张B-scan图像的路径并保存
                    B_scan列表[i] = os.path.join(数据集根目录, 不同大小OCTA, 图像模态, 立体数据名称, B_scan列表[i])
                训练集[不同大小OCTA][图像模态][立体数据名称] = B_scan列表  # 将“B_scan列表”作为键对应的值保存
    # return 训练集
    return 训练集['OCTA_3M'], 训练集['OCTA_6M']


# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class OCTA3D数据集_未关闭(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    """
    def __init__(self, 数据集图像路径字典, 立体数据尺寸:list, 块尺寸:list, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 块尺寸: 一个列表，每个图像块的尺寸，按照深度、高度、宽度排列。
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整型数字。
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        # 高度*宽度*图像数量                                                 270            304          304。这样方便取水平方向上的某一层图像数据
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        if self.立体数据尺寸[1] == 304:
            self.图像数据集合名称 = "Depth64_OCTA_3M_3D_Data"  # 3M对应大小304
        else:
            self.图像数据集合名称 = "Depth64_OCTA_6M_6D_Data"  # 6M对应大小400
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集。")
            # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
            # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            当前数据模态 = -1
            for 数据模态 in self.数据集图像路径字典.values():
                当前数据模态 += 1  # 选中OCT或者OCTA
                当前患者 = -1
                for 每个患者 in 数据模态.values():
                    当前患者 += 1  # 选中一名患者如10001， 10002，...
                    当前B_scan图像 = -1
                    for 图像路径 in 每个患者:
                        当前B_scan图像 += 1  # 选中一张B_scan图像
                        B_scan图像 = np.array(Image.open(图像路径))
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[300:556, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                数据集合文件.create_dataset(self.图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集生成完毕。")
        else:
            数据集合文件 = h5py.File(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5'), "r")
            self.全部图像数据集合 = 数据集合文件[self.图像数据集合名称]
            # 数据集合文件.close()
            print("OCTA图像数据集已存在！")

    def __len__(self):
        if self.立体数据尺寸[1] == 304:
            # 3M数据200组
            return 200
        else:
            # 6M数据300组
            return 300

    def __getitem__(self, item):
        图像块列表 = [] # 用来保存每名患者划分之后的图像块，数据形状[361, 2, 256, 16, 16]
        if self.立体数据尺寸[1] == 304:
            for y起点 in range(0, self.立体数据尺寸[1], self.块尺寸[1]):
                for x起点 in range(0, self.立体数据尺寸[1], self.块尺寸[1]):
                    # 图像块形状[2, 256, 16, 16]
                    图像块 = self.全部图像数据集合[item, :, :, y起点:y起点 + 16, x起点:x起点 + 16].astype(np.float32)
                    图像块列表.append(图像块)
        return 图像块列表



# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class 低显存OCTA3D数据集(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    """
    def __init__(self, 数据集图像路径字典, 立体数据尺寸:list, 块尺寸:list, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 块尺寸: 一个列表，每个图像块的尺寸，按照深度、高度、宽度排列。
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整型数字。
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        # 高度*宽度*图像数量                                                 270            304          304。这样方便取水平方向上的某一层图像数据
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        if self.立体数据尺寸[1] == 304:
            self.图像数据集合名称 = "Depth64_OCTA_3M_3D_Data"  # 3M对应大小304
        else:
            self.图像数据集合名称 = "Depth64_OCTA_6M_6D_Data"  # 6M对应大小400
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集。")
            # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
            # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            当前数据模态 = -1
            for 数据模态 in self.数据集图像路径字典.values():
                当前数据模态 += 1  # 选中OCT或者OCTA
                当前患者 = -1
                for 每个患者 in 数据模态.values():
                    当前患者 += 1  # 选中一名患者如10001， 10002，...
                    当前B_scan图像 = -1
                    for 图像路径 in 每个患者:
                        当前B_scan图像 += 1  # 选中一张B_scan图像
                        B_scan图像 = np.array(Image.open(图像路径))
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[300:556, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                数据集合文件.create_dataset(self.图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集生成完毕。")
        else:
            print("OCTA图像数据集已存在！")

    def 划分立体图像(self, 病人索引, 立体图像数据):
        # TODO 目前在高度方向上没有划分，后续进行transformer可能需要高度上划分
        # 图像块数量 = (self.立体数据尺寸[1] // self.块尺寸[1]) ** 2
        图像块列表 = [] # 以304为例 每名患者数据划分后形状[361, 2, 256, 16, 16]
        if self.立体数据尺寸[1] == 304:
            # 位置 = -1
            for y起点 in range(0, self.立体数据尺寸[1], self.块尺寸[1]):
                for x起点 in range(0, self.立体数据尺寸[1], self.块尺寸[1]):
                    # 位置 += 1
                    图像块 = 立体图像数据[病人索引, :, :, y起点:y起点 + 16, x起点:x起点 + 16].astype(np.float32)
                    图像块列表.append(图像块)

        return 图像块列表

    def __len__(self):
        if self.立体数据尺寸[1] == 304:
            # 3M数据200组
            return 200
        else:
            # 6M数据300组
            return 300

    def __getitem__(self, item):
        # 根据给定索引选中对应患者的立体图像数据，然后划分成16个图像块
        if not os.path.exists(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5')):
            # 如果不存在hdf5数据集
            return self.划分立体图像(item, self.全部图像数据集合)
        else:
            with h5py.File(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5'), "r") as 数据集合文件:
                self.全部图像数据集合 = 数据集合文件[self.图像数据集合名称]
                return self.划分立体图像(item, self.全部图像数据集合)


# '''
if __name__ == '__main__':
    # """
    OCTA_3M图像路径字典, OCTA_6M图像路径字典 = 读数据集图像路径('../Dataset/UnlabeledTrainDataset')
    OCTA立体数据尺寸, 图像块尺寸 = [256, 304, 304], [256, 76, 76]
    # 图像块数量, OCTA模态类别, 患者样本数量, 数据保存路径 = 16, 2, 200, '../Dataset'
    OCTA模态类别, 患者样本数量, 数据保存路径 = 2, 200, '../Dataset'
    # 立体数据集 = hdf5数据集(OCTA_3M图像路径字典, 患者样本数量,OCTA模态类别, OCTA立体数据尺寸, 数据保存路径)


    OCTA3M数据集 = OCTA3D数据集_未关闭(OCTA_3M图像路径字典, OCTA立体数据尺寸, 图像块尺寸, OCTA模态类别, 患者样本数量, 数据保存路径)
    立体图像训练数据 = DataLoader(OCTA3M数据集, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    print(len(立体图像训练数据))
    print(1)
    # """
    """
    OCTA_3M图像路径字典, OCTA_6M图像路径字典 = 读数据集图像路径('../Dataset/UnlabeledTrainDataset')
    OCTA立体数据尺寸, 图像块尺寸 = [256, 304, 304], [256, 76, 76]
    # 图像块数量, OCTA模态类别, 患者样本数量, 数据保存路径 = 16, 2, 200, '../Dataset'
    OCTA模态类别, 患者样本数量, 数据保存路径 = 2, 200, '../Dataset'
    立体数据集 = hdf5数据集(OCTA_3M图像路径字典, 患者样本数量,OCTA模态类别, OCTA立体数据尺寸, 数据保存路径)
    print(立体数据集.close())
    OCTA3M数据集 = OCTA3D数据集(OCTA立体数据尺寸, 图像块尺寸, 立体数据集)
    立体图像训练数据 = DataLoader(OCTA3M数据集, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    print(1)
    """
# '''