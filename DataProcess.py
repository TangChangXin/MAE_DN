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
    # 训练集字典中保存的是每张图像的路径，并不是真的读取了
    # return 训练集
    return 训练集['OCTA_3M'], 训练集['OCTA_6M']


def hdf5数据集(数据集图像路径字典, 患者数量, 图像模态种类, 立体数据尺寸, 保存路径):
    """
    根据输入的数据集图像路径字典，读取图像并生成hdf5形式的数据集，然后返回hdf5形式的数据集；若数据集已存在则读取数据集后返回。
    :param 数据集图像路径字典:
    :param 患者数量:
    :param 图像模态种类:
    :param 立体数据尺寸:
    :param 保存路径:
    :return: 返回hdf5形式的数据集
    """
    if 立体数据尺寸[1] == 304:
        图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
    else:
        图像数据集合名称 = "OCTA_6M_3D_Data"  # 6M对应大小400
    # 如果不存在数据。
    全部图像数据集合 = np.zeros((患者数量, 图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
    if not os.path.exists(os.path.join(保存路径, 图像数据集合名称 + '.hdf5')):
        print("生成OCTA图像数据集合。")
        # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
        # 如果数据集不考虑
        当前数据模态 = -1
        for 数据模态 in 数据集图像路径字典.values():
            当前数据模态 += 1  # 选中OCT或者OCTA
            当前患者 = -1
            for 每个患者 in 数据模态.values():
                当前患者 += 1  # 选中一名患者如10001， 10002，...
                当前B_scan图像 = -1
                for 图像路径 in 每个患者:
                    当前B_scan图像 += 1  # 选中一张B_scan图像
                    B_scan图像 = np.array(Image.open(图像路径))
                    全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

        # 保存hdf5数据集文件
        with h5py.File(os.path.join(保存路径, 图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
            数据集合文件.create_dataset(图像数据集合名称, data=全部图像数据集合, dtype=np.uint8)
        print("OCTA图像数据集合生成完毕。")
    else:
        print("OCTA图像数据集合已存在！")
    # 函数调用后未关闭，后续应该关闭
    return h5py.File(os.path.join(保存路径, 图像数据集合名称 + '.hdf5'), "r")



# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度
class OCTA3D数据集1(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    直接传入缩小后的立体数据尺寸，暂时不适用块尺寸
    """

    def __init__(self, 数据集图像路径字典, 立体数据尺寸, 块尺寸, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 块尺寸: 一个列表，
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整形数字，
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        self.迭代测试 = np.linspace(1, 200, 200, endpoint=True, dtype=np.uint8)
        self.图像块 = np.zeros((self.图像模态种类, 块尺寸[0], 块尺寸[1], 块尺寸[2]))  # TODO这才是真正输入网络的数据
        #                                           高度*宽度*图像数量:     240            400          400。这样方便取水平方向上的某一层图像数据
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        self.读OCTA图像()

    def __getitem__(self, item):
        # 根据给定索引返回对应患者的立体图像数据
        立体图像 = self.全部图像数据集合[item]
        立体图像 = torch.as_tensor(立体图像)  # 转换为张量
        当前患者 = self.迭代测试[item]
        return 立体图像, 当前患者

    # 类初始化时传入的数据集图像路径字典
    # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
    # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
    def 读OCTA图像(self):
        """
        类初始化时根据形参“立体数据尺寸”判断读3M还是6M图像。之后读取对应大小的所有OCTA图像，得到的结果保存在”self.全部图像数据集合“中。
        然后以hdf5形式保存为一个数据集合，存储在硬盘中方便后续再次读取时使用。
        """
        # TODO 目前不考虑显存溢出
        if self.立体数据尺寸[1] == 400:
            图像数据集合名称 = "OCTA_6M_3D_Data"  # 6M对应大小400
        else:
            图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集合。")
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            # 如果数据集不考虑
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
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                # todo 生成数据集的名字和文件的名字一样是否有问题
                数据集合文件.create_dataset(图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集合生成完毕。")
        else:
            print("OCTA图像数据集合已存在！")
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "r") as 数据集合文件:
                self.全部图像数据集合 = 数据集合文件[图像数据集合名称]

    # def 读取批量数据(self):
    #     """
    #     TODO 如何从数据集合中取出每小块数据
    #
    #     :return:
    #     """
    #     # 如何实现自动批量迭代数据。400张图像构成一组图像，一个批次中包含若干组图像
    #     for 每组图像 in range(0, self.批量大小):
    #         self.图像块[每组图像, :, :, :, :] = self.全部图像数据集合[]


# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class OCTA3D数据集2(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    直接传入缩小后的立体数据尺寸，暂时不适用块尺寸
    """

    def __init__(self, 数据集图像路径字典, 立体数据尺寸, 块尺寸, 图像视角种类, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 块尺寸: 一个列表，
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整形数字，
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.块尺寸 = 块尺寸
        self.图像视角种类 = 图像视角种类
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        # self.批量大小 = 批量大小
        self.保存路径 = 保存路径
        # todo 图像块需要考虑OCTA大小区别吗
        # TODO 假设显存溢出了，如何划分图像块
        self.迭代测试 = np.linspace(1, 200, 200, endpoint=True, dtype=np.uint8)
        self.图像块 = np.zeros((self.图像模态种类, 块尺寸[0], 块尺寸[1], 块尺寸[2]))  # TODO这才是真正输入网络的数据
        # self.全部图像数据集合 = np.zeros((self.图像模态种类, 块尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2], self.患者数量), dtype=np.uint8)
        # 高度*宽度*图像数量                                                     240            400          400。这样方便取水平方向上的某一层图像数据
        # self.全部图像数据集合 = np.zeros((self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2], self.患者数量),
        #                          dtype=np.uint8)
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.图像视角种类, self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        # self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        self.读OCTA图像()

    def __getitem__(self, item):
        # 根据给定索引返回对应患者的立体图像数据
        立体图像 = self.全部图像数据集合[item]
        立体图像 = torch.as_tensor(立体图像)  # 转换为张量
        当前患者 = self.迭代测试[item]
        return 立体图像, 当前患者

    # 类初始化时传入的数据集图像路径字典
    # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
    # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
    def 读OCTA图像(self):
        """
        类初始化时根据形参“立体数据尺寸”判断读3M还是6M图像。之后读取对应大小的所有OCTA图像，得到的结果保存在”self.全部图像数据集合“中。
        然后以hdf5形式保存为一个数据集合，存储在硬盘中方便后续再次读取时使用。
        """
        # TODO 目前不考虑显存溢出
        if self.立体数据尺寸[1] == 400:
            图像数据集合名称 = "OCTA_6M_3D_Data"  # 6M对应大小400
        else:
            图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集合。")
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            # 如果数据集不考虑
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
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                # todo 生成数据集的名字和文件的名字一样是否有问题
                数据集合文件.create_dataset(图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集合生成完毕。")
        else:
            print("OCTA图像数据集合已存在！")
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "r") as 数据集合文件:
                self.全部图像数据集合 = 数据集合文件[图像数据集合名称]

    # def 读取批量数据(self):
    #     """
    #     TODO 如何从数据集合中取出每小块数据
    #
    #     :return:
    #     """
    #     # 如何实现自动批量迭代数据。400张图像构成一组图像，一个批次中包含若干组图像
    #     for 每组图像 in range(0, self.批量大小):
    #         self.图像块[每组图像, :, :, :, :] = self.全部图像数据集合[]


# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class OCTA3D数据集3(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    直接传入缩小后的立体数据尺寸，暂时不适用块尺寸
    """

    def __init__(self, 数据集图像路径字典, 立体数据尺寸, 图像块数量, 块尺寸, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 图像块数量: 将原始立体图像数据按照给定数量划分为若干图像块。
        :param 块尺寸: 一个列表，
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整形数字，
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.图像块数量 = 图像块数量
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        self.迭代测试 = np.linspace(1, 200, 200, endpoint=True, dtype=np.uint8)
        self.图像块 = np.zeros((self.图像块数量, self.图像模态种类, 块尺寸[0], 块尺寸[1], 块尺寸[2]), dtype=np.uint8)  # TODO这才是真正输入网络的数据
        # 高度*宽度*图像数量                                                     240            400          400。这样方便取水平方向上的某一层图像数据
        # self.全部图像数据集合 = np.zeros((self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2], self.患者数量), dtype=np.uint8)
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        self.读OCTA图像()

    def __len__(self):
        if self.立体数据尺寸[1] == 304:
            # 3M数据200组
            return 200
        else:
            # 6M数据300组
            return 300
        # return len(self.全部图像数据集合.shape[0])

    def __getitem__(self, item):
        """
        # 根据给定索引返回对应患者的立体图像数据。这种方式显存溢出
        立体图像 = self.全部图像数据集合[item]
        立体图像 = torch.as_tensor(立体图像) # 转换为张量
        当前患者 = self.迭代测试[item]
        return 立体图像, 当前患者
        """



        # 根据给定索引返回对应患者的立体图像数据，然后划分成16个图像块
        # 立体图像 = self.全部图像数据集合[item]
        """
        if self.立体数据尺寸[1] == 304:
            # 3M数据的形状                      C  D    H    W
            图像块1 = self.全部图像数据集合[item, :, :, 0:75, 0:75]
            图像块2 = self.全部图像数据集合[item, :, :, 0:75, 76:151]
            图像块3 = self.全部图像数据集合[item, :, :, 0:75, 152:227]
            图像块4 = self.全部图像数据集合[item, :, :, 0:75, 228:303]
            图像块5 = self.全部图像数据集合[item, :, :, 76:151, 0:75]
            图像块6 = self.全部图像数据集合[item, :, :, 76:151, 76:151]
            图像块7 = self.全部图像数据集合[item, :, :, 76:151, 152:227]
            图像块8 = self.全部图像数据集合[item, :, :, 76:151, 228:303]
            图像块9 = self.全部图像数据集合[item, :, :, 152:227, 0:75]
            图像块10 = self.全部图像数据集合[item, :, :, 152:227, 76:151]
            图像块11 = self.全部图像数据集合[item, :, :, 152:227, 152:227]
            图像块12 = self.全部图像数据集合[item, :, :, 152:227, 228:303]
            图像块13 = self.全部图像数据集合[item, :, :, 228:303, 0:75]
            图像块14 = self.全部图像数据集合[item, :, :, 228:303, 76:151]
            图像块15 = self.全部图像数据集合[item, :, :, 228:303, 152:227]
            图像块16 = self.全部图像数据集合[item, :, :, 228:303, 228:303]

            # 图像块1 = torch.as_tensor(图像块1, dtype=torch.uint8)
            # 图像块2 = torch.as_tensor(图像块2, dtype=torch.uint8)
            # 图像块3 = torch.as_tensor(图像块3, dtype=torch.uint8)
            # 图像块4 = torch.as_tensor(图像块4, dtype=torch.uint8)
            # 图像块5 = torch.as_tensor(图像块5, dtype=torch.uint8)
            # 图像块6 = torch.as_tensor(图像块6, dtype=torch.uint8)
            # 图像块7 = torch.as_tensor(图像块7, dtype=torch.uint8)
            # 图像块8 = torch.as_tensor(图像块8, dtype=torch.uint8)
            # 图像块9 = torch.as_tensor(图像块9, dtype=torch.uint8)
            # 图像块10 = torch.as_tensor(图像块10, dtype=torch.uint8)
            # 图像块11 = torch.as_tensor(图像块11, dtype=torch.uint8)
            # 图像块12 = torch.as_tensor(图像块12, dtype=torch.uint8)
            # 图像块13 = torch.as_tensor(图像块13, dtype=torch.uint8)
            # 图像块14 = torch.as_tensor(图像块14, dtype=torch.uint8)
            # 图像块15 = torch.as_tensor(图像块15, dtype=torch.uint8)
            # 图像块16 = torch.as_tensor(图像块16, dtype=torch.uint8)
        else:
            # 6M数据的形状                      C  D    H    W
            图像块1 = self.全部图像数据集合[item, :, :, 0:75, 0:75]
            图像块2 = self.全部图像数据集合[item, :, :, 0:75, 76:151]
            图像块3 = self.全部图像数据集合[item, :, :, 0:75, 152:227]
            图像块4 = self.全部图像数据集合[item, :, :, 0:75, 228:303]
            图像块5 = self.全部图像数据集合[item, :, :, 76:151, 0:75]
            图像块6 = self.全部图像数据集合[item, :, :, 76:151, 76:151]
            图像块7 = self.全部图像数据集合[item, :, :, 76:151, 152:227]
            图像块8 = self.全部图像数据集合[item, :, :, 76:151, 228:303]
            图像块9 = self.全部图像数据集合[item, :, :, 152:227, 0:75]
            图像块10 = self.全部图像数据集合[item, :, :, 152:227, 76:151]
            图像块11 = self.全部图像数据集合[item, :, :, 152:227, 152:227]
            图像块12 = self.全部图像数据集合[item, :, :, 152:227, 228:303]
            图像块13 = self.全部图像数据集合[item, :, :, 228:303, 0:75]
            图像块14 = self.全部图像数据集合[item, :, :, 228:303, 76:151]
            图像块15 = self.全部图像数据集合[item, :, :, 228:303, 152:227]
            图像块16 = self.全部图像数据集合[item, :, :, 228:303, 228:303]

            图像块1 = torch.as_tensor(图像块1, dtype=torch.uint8)
            图像块2 = torch.as_tensor(图像块2, dtype=torch.uint8)
            图像块3 = torch.as_tensor(图像块3, dtype=torch.uint8)
            图像块4 = torch.as_tensor(图像块4, dtype=torch.uint8)
            图像块5 = torch.as_tensor(图像块5, dtype=torch.uint8)
            图像块6 = torch.as_tensor(图像块6, dtype=torch.uint8)
            图像块7 = torch.as_tensor(图像块7, dtype=torch.uint8)
            图像块8 = torch.as_tensor(图像块8, dtype=torch.uint8)
            图像块9 = torch.as_tensor(图像块9, dtype=torch.uint8)
            图像块10 = torch.as_tensor(图像块10, dtype=torch.uint8)
            图像块11 = torch.as_tensor(图像块11, dtype=torch.uint8)
            图像块12 = torch.as_tensor(图像块12, dtype=torch.uint8)
            图像块13 = torch.as_tensor(图像块13, dtype=torch.uint8)
            图像块14 = torch.as_tensor(图像块14, dtype=torch.uint8)
            图像块15 = torch.as_tensor(图像块15, dtype=torch.uint8)
            图像块16 = torch.as_tensor(图像块16, dtype=torch.uint8)
        """

        return self.迭代测试[item]
        # return 图像块1, 图像块2, 图像块3, 图像块4, 图像块5, 图像块6, 图像块7, 图像块8,\
        #        图像块9, 图像块10, 图像块11, 图像块12, 图像块13, 图像块14, 图像块15, 图像块16

    # 类初始化时传入的数据集图像路径字典
    # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
    # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
    def 读OCTA图像(self):
        """
        类初始化时根据形参“立体数据尺寸”判断读3M还是6M图像。之后读取对应大小的所有OCTA图像，得到的结果保存在”self.全部图像数据集合“中。
        然后以hdf5形式保存为一个数据集合，存储在硬盘中方便后续再次读取时使用。
        """
        # TODO 目前不考虑显存溢出
        if self.立体数据尺寸[1] == 304:
            图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
        else:
            图像数据集合名称 = "OCTA_6M_6D_Data"  # 6M对应大小400
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集合。")
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            # 如果数据集不考虑
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
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                # todo 生成数据集的名字和文件的名字一样是否有问题
                数据集合文件.create_dataset(图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集合生成完毕。")
        else:
            print("OCTA图像数据集合已存在！")
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "r") as 数据集合文件:
                self.全部图像数据集合 = 数据集合文件['OCTA_3M_3D_Data']


# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class OCTA3D数据集4(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    直接传入缩小后的立体数据尺寸，暂时不适用块尺寸
    """

    def __init__(self, 数据集图像路径字典, 立体数据尺寸, 图像块数量, 块尺寸, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 图像块数量: 将原始立体图像数据按照给定数量划分为若干图像块。
        :param 块尺寸: 一个列表，
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整形数字，
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        self.图像块数量 = 图像块数量
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类
        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        self.迭代测试 = np.linspace(1, 200, 200, endpoint=True, dtype=np.uint8)
        self.图像块 = np.zeros((self.图像块数量, self.图像模态种类, 块尺寸[0], 块尺寸[1], 块尺寸[2]), dtype=np.uint8)  # TODO这才是真正输入网络的数据
        # 高度*宽度*图像数量                                                     240            400          400。这样方便取水平方向上的某一层图像数据
        # self.全部图像数据集合 = np.zeros((self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2], self.患者数量), dtype=np.uint8)
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        # self.读OCTA图像()

    def __len__(self):
        if self.立体数据尺寸[1] == 304:
            # 3M数据200组
            return 200
        else:
            # 6M数据300组
            return 300
        # return len(self.全部图像数据集合.shape[0])

    def __getitem__(self, item):
        """
        # 根据给定索引返回对应患者的立体图像数据。这种方式显存溢出
        立体图像 = self.全部图像数据集合[item]
        立体图像 = torch.as_tensor(立体图像) # 转换为张量
        当前患者 = self.迭代测试[item]
        return 立体图像, 当前患者
        """

        if self.立体数据尺寸[1] == 304:
            图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
        else:
            图像数据集合名称 = "OCTA_6M_6D_Data"  # 6M对应大小400
        # 如果不存在数据。
        if not os.path.exists(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5')):
            print("生成OCTA图像数据集合。")
            # 所有索引初始化为-1，for循环开始后加1就是从0开始了，后续就按照1,2,3,...遍历了
            # 如果数据集不考虑
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
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                # todo 生成数据集的名字和文件的名字一样是否有问题
                数据集合文件.create_dataset(图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集合生成完毕。")
        else:
            print("OCTA图像数据集合已存在！")
            数据集合文件 = h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "r")
            self.全部图像数据集合 = 数据集合文件['OCTA_3M_3D_Data']
            # 数据集合文件.close()


            # with h5py.File(os.path.join(self.保存路径, 图像数据集合名称 + '.hdf5'), "r") as 数据集合文件:
            #     # self.全部图像数据集合 = 数据集合文件['OCTA_3M_3D_Data']
            #     # print(type(self.全部图像数据集合))
            #
            #     全部图像数据集合 = 数据集合文件['OCTA_3M_3D_Data']
            #     print(type(全部图像数据集合))




        # 根据给定索引返回对应患者的立体图像数据，然后划分成16个图像块
        # 立体图像 = self.全部图像数据集合[item]
        # """
        if self.立体数据尺寸[1] == 304:
            # 3M数据的形状                      C  D    H    W
            图像块1 = self.全部图像数据集合[item, :, :, 0:75, 0:75]
            图像块2 = self.全部图像数据集合[item, :, :, 0:75, 76:151]
            图像块3 = self.全部图像数据集合[item, :, :, 0:75, 152:227]
            图像块4 = self.全部图像数据集合[item, :, :, 0:75, 228:303]
            图像块5 = self.全部图像数据集合[item, :, :, 76:151, 0:75]
            图像块6 = self.全部图像数据集合[item, :, :, 76:151, 76:151]
            图像块7 = self.全部图像数据集合[item, :, :, 76:151, 152:227]
            图像块8 = self.全部图像数据集合[item, :, :, 76:151, 228:303]
            图像块9 = self.全部图像数据集合[item, :, :, 152:227, 0:75]
            图像块10 = self.全部图像数据集合[item, :, :, 152:227, 76:151]
            图像块11 = self.全部图像数据集合[item, :, :, 152:227, 152:227]
            图像块12 = self.全部图像数据集合[item, :, :, 152:227, 228:303]
            图像块13 = self.全部图像数据集合[item, :, :, 228:303, 0:75]
            图像块14 = self.全部图像数据集合[item, :, :, 228:303, 76:151]
            图像块15 = self.全部图像数据集合[item, :, :, 228:303, 152:227]
            图像块16 = self.全部图像数据集合[item, :, :, 228:303, 228:303]

            # 图像块1 = torch.as_tensor(图像块1, dtype=torch.uint8)
            # 图像块2 = torch.as_tensor(图像块2, dtype=torch.uint8)
            # 图像块3 = torch.as_tensor(图像块3, dtype=torch.uint8)
            # 图像块4 = torch.as_tensor(图像块4, dtype=torch.uint8)
            # 图像块5 = torch.as_tensor(图像块5, dtype=torch.uint8)
            # 图像块6 = torch.as_tensor(图像块6, dtype=torch.uint8)
            # 图像块7 = torch.as_tensor(图像块7, dtype=torch.uint8)
            # 图像块8 = torch.as_tensor(图像块8, dtype=torch.uint8)
            # 图像块9 = torch.as_tensor(图像块9, dtype=torch.uint8)
            # 图像块10 = torch.as_tensor(图像块10, dtype=torch.uint8)
            # 图像块11 = torch.as_tensor(图像块11, dtype=torch.uint8)
            # 图像块12 = torch.as_tensor(图像块12, dtype=torch.uint8)
            # 图像块13 = torch.as_tensor(图像块13, dtype=torch.uint8)
            # 图像块14 = torch.as_tensor(图像块14, dtype=torch.uint8)
            # 图像块15 = torch.as_tensor(图像块15, dtype=torch.uint8)
            # 图像块16 = torch.as_tensor(图像块16, dtype=torch.uint8)
        else:
            # 6M数据的形状                      C  D    H    W
            图像块1 = self.全部图像数据集合[item, :, :, 0:75, 0:75]
            图像块2 = self.全部图像数据集合[item, :, :, 0:75, 76:151]
            图像块3 = self.全部图像数据集合[item, :, :, 0:75, 152:227]
            图像块4 = self.全部图像数据集合[item, :, :, 0:75, 228:303]
            图像块5 = self.全部图像数据集合[item, :, :, 76:151, 0:75]
            图像块6 = self.全部图像数据集合[item, :, :, 76:151, 76:151]
            图像块7 = self.全部图像数据集合[item, :, :, 76:151, 152:227]
            图像块8 = self.全部图像数据集合[item, :, :, 76:151, 228:303]
            图像块9 = self.全部图像数据集合[item, :, :, 152:227, 0:75]
            图像块10 = self.全部图像数据集合[item, :, :, 152:227, 76:151]
            图像块11 = self.全部图像数据集合[item, :, :, 152:227, 152:227]
            图像块12 = self.全部图像数据集合[item, :, :, 152:227, 228:303]
            图像块13 = self.全部图像数据集合[item, :, :, 228:303, 0:75]
            图像块14 = self.全部图像数据集合[item, :, :, 228:303, 76:151]
            图像块15 = self.全部图像数据集合[item, :, :, 228:303, 152:227]
            图像块16 = self.全部图像数据集合[item, :, :, 228:303, 228:303]

            # 图像块1 = torch.as_tensor(图像块1, dtype=torch.uint8)
            # 图像块2 = torch.as_tensor(图像块2, dtype=torch.uint8)
            # 图像块3 = torch.as_tensor(图像块3, dtype=torch.uint8)
            # 图像块4 = torch.as_tensor(图像块4, dtype=torch.uint8)
            # 图像块5 = torch.as_tensor(图像块5, dtype=torch.uint8)
            # 图像块6 = torch.as_tensor(图像块6, dtype=torch.uint8)
            # 图像块7 = torch.as_tensor(图像块7, dtype=torch.uint8)
            # 图像块8 = torch.as_tensor(图像块8, dtype=torch.uint8)
            # 图像块9 = torch.as_tensor(图像块9, dtype=torch.uint8)
            # 图像块10 = torch.as_tensor(图像块10, dtype=torch.uint8)
            # 图像块11 = torch.as_tensor(图像块11, dtype=torch.uint8)
            # 图像块12 = torch.as_tensor(图像块12, dtype=torch.uint8)
            # 图像块13 = torch.as_tensor(图像块13, dtype=torch.uint8)
            # 图像块14 = torch.as_tensor(图像块14, dtype=torch.uint8)
            # 图像块15 = torch.as_tensor(图像块15, dtype=torch.uint8)
            # 图像块16 = torch.as_tensor(图像块16, dtype=torch.uint8)
        # """

        # return self.迭代测试[item]
        return 图像块1, 图像块2, 图像块3, 图像块4, 图像块5, 图像块6, 图像块7, 图像块8,\
               图像块9, 图像块10, 图像块11, 图像块12, 图像块13, 图像块14, 图像块15, 图像块16


    # 类初始化时传入的数据集图像路径字典
    # OCTA_3M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 3M的形状是304*640
    # OCTA_6M {'OCT':{'10001':[1.bmp, 2.bmp, ...], ...}, 'OCTA':{'10001':[1.bmp, 2.bmp, ...], ...}} 6M的形状是400*640
    def 读OCTA图像(self):
        """
        类初始化时根据形参“立体数据尺寸”判断读3M还是6M图像。之后读取对应大小的所有OCTA图像，得到的结果保存在”self.全部图像数据集合“中。
        然后以hdf5形式保存为一个数据集合，存储在硬盘中方便后续再次读取时使用。
        """


# 输入网络的数据形式是CDHW，即通道数、深度、高度、宽度。读两个视角测试
class OCTA3D数据集(Dataset):
    """
    读OCTA_3M或者OCTA_6M的3维数据。形参“图像模态种类”理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据
    .. note::
    """

    图像数据集合名称 = ""
    def __init__(self, 数据集图像路径字典, 立体数据尺寸, 块尺寸, 图像模态种类, 患者数量, 保存路径):
        """

        :param 数据集图像路径字典: 传入的字典中只包含3M图像或者6M图像的路径，不能同时包含两者。
        :param 立体数据尺寸: 一个列表，指定每个患者的3维OCTA数据的尺寸，按照深度、高度、宽度排列。
        :param 图像块数量: 将原始立体图像数据按照给定数量划分为若干图像块。
        :param 块尺寸: 一个列表，每个图像块的尺寸。
        :param 图像模态种类: 理解为输入图像的通道数，灰度图实际单通道，将OCTA和OCT分别输入模型训练视为用两种通道的数据。
            同时使用OCTA和OCT两种模态的话是2，只使用一种模态的话是1。
        :param 患者数量: 整型数字。
        :param 保存路径: HDF5形式数据集合保存的位置。
        """
        self.数据集图像路径字典 = 数据集图像路径字典
        self.立体数据尺寸 = 立体数据尺寸
        # self.图像块数量 = 图像块数量
        self.块尺寸 = 块尺寸
        self.图像模态种类 = 图像模态种类

        self.患者数量 = 患者数量
        self.保存路径 = 保存路径
        # 高度*宽度*图像数量                                                 240            400          400。这样方便取水平方向上的某一层图像数据
        # CDHW，即通道数、深度、高度、宽度                         C            D              H             W
        self.全部图像数据集合 = np.zeros((self.患者数量, self.图像模态种类, 立体数据尺寸[0], 立体数据尺寸[1], 立体数据尺寸[2]), dtype=np.uint8)
        if self.立体数据尺寸[1] == 304:
            self.图像数据集合名称 = "OCTA_3M_3D_Data"  # 3M对应大小304
        else:
            self.图像数据集合名称 = "OCTA_6M_6D_Data"  # 6M对应大小400
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
                        self.全部图像数据集合[当前患者, 当前数据模态, :, :, 当前B_scan图像] = B_scan图像[200:440, :]

            # 保存数据集合文件
            with h5py.File(os.path.join(self.保存路径, self.图像数据集合名称 + '.hdf5'), "w") as 数据集合文件:
                # todo 生成数据集的名字和文件的名字一样是否有问题
                数据集合文件.create_dataset(self.图像数据集合名称, data=self.全部图像数据集合, dtype=np.uint8)
            print("OCTA图像数据集生成完毕。")
        else:
            print("OCTA图像数据集已存在！")


    def 划分立体图像(self, 病人索引, 立体图像数据):
        # TODO 目前在高度方向上没有划分，后续进行transformer可能需要高度上划分
        # 图像排列的时候，第一张对应W维度的0，然后依次排列，最后一张对应303或者399。
        # H维度对应的是原二维图像的宽度。
        if self.立体数据尺寸[1] == 304:
            # 3M数据的形状                C  D    H    W
            图像块1 = 立体图像数据[病人索引, :, :, 0:75, 0:75]
            图像块2 = 立体图像数据[病人索引, :, :, 0:75, 76:151]
            图像块3 = 立体图像数据[病人索引, :, :, 0:75, 152:227]
            图像块4 = 立体图像数据[病人索引, :, :, 0:75, 228:303]
            图像块5 = 立体图像数据[病人索引, :, :, 76:151, 0:75]
            图像块6 = 立体图像数据[病人索引, :, :, 76:151, 76:151]
            图像块7 = 立体图像数据[病人索引, :, :, 76:151, 152:227]
            图像块8 = 立体图像数据[病人索引, :, :, 76:151, 228:303]
            图像块9 = 立体图像数据[病人索引, :, :, 152:227, 0:75]
            图像块10 = 立体图像数据[病人索引, :, :, 152:227, 76:151]
            图像块11 = 立体图像数据[病人索引, :, :, 152:227, 152:227]
            图像块12 = 立体图像数据[病人索引, :, :, 152:227, 228:303]
            图像块13 = 立体图像数据[病人索引, :, :, 228:303, 0:75]
            图像块14 = 立体图像数据[病人索引, :, :, 228:303, 76:151]
            图像块15 = 立体图像数据[病人索引, :, :, 228:303, 152:227]
            图像块16 = 立体图像数据[病人索引, :, :, 228:303, 228:303]
        else:
            # 6M数据的形状                C  D    H    W
            图像块1 = 立体图像数据[病人索引, :, :, 0:75, 0:75]
            图像块2 = 立体图像数据[病人索引, :, :, 0:75, 76:151]
            图像块3 = 立体图像数据[病人索引, :, :, 0:75, 152:227]
            图像块4 = 立体图像数据[病人索引, :, :, 0:75, 228:303]
            图像块5 = 立体图像数据[病人索引, :, :, 76:151, 0:75]
            图像块6 = 立体图像数据[病人索引, :, :, 76:151, 76:151]
            图像块7 = 立体图像数据[病人索引, :, :, 76:151, 152:227]
            图像块8 = 立体图像数据[病人索引, :, :, 76:151, 228:303]
            图像块9 = 立体图像数据[病人索引, :, :, 152:227, 0:75]
            图像块10 = 立体图像数据[病人索引, :, :, 152:227, 76:151]
            图像块11 = 立体图像数据[病人索引, :, :, 152:227, 152:227]
            图像块12 = 立体图像数据[病人索引, :, :, 152:227, 228:303]
            图像块13 = 立体图像数据[病人索引, :, :, 228:303, 0:75]
            图像块14 = 立体图像数据[病人索引, :, :, 228:303, 76:151]
            图像块15 = 立体图像数据[病人索引, :, :, 228:303, 152:227]
            图像块16 = 立体图像数据[病人索引, :, :, 228:303, 228:303]

        return 图像块1, 图像块2, 图像块3, 图像块4, 图像块5, 图像块6, 图像块7, 图像块8, \
               图像块9, 图像块10, 图像块11, 图像块12, 图像块13, 图像块14, 图像块15, 图像块16


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
                self.全部图像数据集合 = 数据集合文件['OCTA_3M_3D_Data']
                return self.划分立体图像(item, self.全部图像数据集合)



# '''
if __name__ == '__main__':
    OCTA_3M图像路径字典, OCTA_6M图像路径字典 = 读数据集图像路径('../Dataset/UnlabeledTrainDataset')
    OCTA立体数据尺寸, 图像块尺寸 = [240, 304, 304], [240, 76, 76]
    # 图像块数量, OCTA模态类别, 患者样本数量, 数据保存路径 = 16, 2, 200, '../Dataset'
    OCTA模态类别, 患者样本数量, 数据保存路径 = 2, 200, '../Dataset'
    # 立体数据集 = hdf5数据集(OCTA_3M图像路径字典, 患者样本数量,OCTA模态类别, OCTA立体数据尺寸, 数据保存路径)


    OCTA3M数据集 = OCTA3D数据集(OCTA_3M图像路径字典, OCTA立体数据尺寸, 图像块尺寸, OCTA模态类别, 患者样本数量, 数据保存路径)
    立体图像训练数据 = DataLoader(OCTA3M数据集, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    print(len(立体图像训练数据))
    print(1)
# '''