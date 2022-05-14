import sys, os, math, argparse, torch, random
# import torch.optim as optim
# import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.backends.cudnn import deterministic
import numpy as np
import DataProcess
import model


# from my_dataset import MyDataSet
# from vit_model import vit_base_patch16_224_in21k as create_model
# from utils import read_split_data, evaluate


# 先用3M的数据训练，中间保存最好的模型，然后也保存训练结束的模型，再输入6M数据继续训练。
# 先用6M的数据训练，中间保存最好的模型，然后也保存训练结束的模型，再输入3M数据继续训练。

# 设置一个参数解析器
命令行参数解析器 = argparse.ArgumentParser(description='使用3维数据进行自监督训练，之后再用2维图像微调')
# 添加无标签数据训练时的参数
命令行参数解析器.add_argument('--max_epoch', type=int, default=6000, help="无标签训练的最大迭代周期")
命令行参数解析器.add_argument('--method', type=str, default='conv', help='融合立体图像时使用的结构') # TODO 需要修改的命令行参数
# 数据和块大小以3M数据为例
命令行参数解析器.add_argument('--volume_data_size', type=list, default=[256, 304, 304], help='立体图像数据的3维形状，按照深度，高度，宽度的顺序排列，即D*H*W')
命令行参数解析器.add_argument('--block_size', type=list, default=[256, 16, 16], help='划分成图像块的形状，按照深度，高度，宽度的顺序排列，即D*H*W')
命令行参数解析器.add_argument('--input_modalities', type=int, default=2, help='输入的图像模态种类，OCTA和OCT数据都用就是2，只用一种则是1')

# 获取命令行传入的参数
无标签训练命令行参数 = 命令行参数解析器.parse_args()

全部随机数种子 = 222  # init seed 初始化随机种子
# 下面似乎都是控制生成相同的随机数
random.seed(全部随机数种子)
np.random.seed(全部随机数种子)
torch.manual_seed(全部随机数种子)
torch.cuda.manual_seed_all(全部随机数种子)
torch.cuda.manual_seed(全部随机数种子)
np.random.seed(全部随机数种子)
os.environ['PYTHONHASHSEED'] = str(全部随机数种子)  # 禁止哈希随机化，使实验可复现

# 设置训练使用的设备
# '''
if torch.cuda.is_available():
    硬件设备 = torch.device("cuda:0")
    # 保证每次返回的卷积算法将是确定的，如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # 为每层搜索适合的卷积算法实现，加速计算
else:
    硬件设备 = torch.device("cpu")
# '''
# 硬件设备 = torch.device("cpu")
print("训练使用设备", 硬件设备)

def 训练模型(网络模型, 优化器, 硬件设备, 命令行参数):
    OCTA_3M图像路径字典, OCTA_6M图像路径字典 = DataProcess.读数据集图像路径('../Dataset/UnlabeledTrainDataset')
    # OCTA立体数据尺寸, 图像块尺寸 = [240, 304, 304], [240, 76, 76]
    OCTA立体数据尺寸, 图像块尺寸 = 命令行参数.volume_data_size, 命令行参数.block_size
    # 图像块数量, OCTA模态类别, 患者样本数量, 数据保存路径 = 16, 2, 200, '../Dataset'
    OCTA模态类别, 患者样本数量, 数据保存路径 = 命令行参数.input_modalities, 200, '../Dataset'
    # TODO 如何同时传入两个字典，但是在类的内部初始化的时候区分，并在后续使用的时候区分
    # OCTA3M数据集 = DataProcess.低显存OCTA3D数据集(OCTA_3M图像路径字典, OCTA立体数据尺寸, 图像块尺寸, OCTA模态类别, 患者样本数量, 数据保存路径)
    OCTA3M数据集 = DataProcess.OCTA3D数据集_未关闭(OCTA_3M图像路径字典, OCTA立体数据尺寸, 图像块尺寸, OCTA模态类别, 患者样本数量, 数据保存路径)
    立体图像训练数据 = DataProcess.DataLoader(OCTA3M数据集, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # TODO 可能需要两个不同的损失函数。虽然都是mse
    # 重建立体图像损失函数, 重建融合图像损失函数 = 命令行参数.loss1, 命令行参数.loss2

    损失函数 = torch.nn.MSELoss().to(硬件设备)

    for 当前训练周期 in range(1, 命令行参数.max_epoch):
        网络模型.train() # 开始训练
        当前训练周期全部损失 = 0
        训练循环 = tqdm(enumerate(立体图像训练数据), total=len(立体图像训练数据), ncols=130, colour='#d33682',leave=True)
        # 当前患者索引 = -1
        #    批量   C  D    H    W
        # [病人索引, :, :, 0:76, 0:76]
        # 图像排列的时候，第一张对应W维度的0，然后依次排列，最后一张对应303或者399。H维度对应的是原二维图像的宽度。
        # 目前每名患者划分为16个图像块，每个图像块先后送入模型训练，输出为每个图像块的融合结果，然后将16个融合结果拼起来得到每个病人的立体图像数据的融合结果。
        # 每名患者全部图像块的形状 torch.Size([1, 361, 2, 256, 16, 16])
        for 训练批次, 每名患者全部图像块 in 训练循环:
            每名患者训练损失 = torch.tensor(0, dtype=torch.float32).to(硬件设备) # 每名患者的训练损失初始为张量形式的0，不然调用backward()报错
            # 每名患者的所有图像块算完之后再计算损失并反向传播然后更新。
            # 这里定义一个量用来保存融合图像
            # todo 定义一个列表保存融合图像
            # 拼接图像 = torch.tensor([])
            拼接图像 = []
            for 每个图像块 in 每名患者全部图像块:
                每个图像块 = torch.div(每个图像块, 255) #
                每个图像块 = 每个图像块.to(硬件设备)
                融合图像 = 网络模型(每个图像块)

                # 拼接图像.append(融合图像) # 这里要把输出的融合图像拼接起来，现在不直接拼成一张完整的图，直接以列表形式保存。

                # print(融合图像)
                # 融合图像, 重建图像 = 网络模型(每个图像块) # 输出是

                # 每名患者训练损失 += 损失函数(重建图像, 每个图像块)  # 图像块的损失累加

                # 融合图像 = 网络模型(每个图像块) # 输出是
                # 每名患者训练损失 += 损失函数(融合图像, 每个图像块) # 图像块的损失累加

            # print(1)
            # 拼接图像.to(硬件设备)

            # TODO 如何拼接输出的融合图像构成一张OCTA图像，作为后续自监督的输入
            # 优化器.zero_grad()
            # 每名患者训练损失.backward()
            # 优化器.step()
            # 当前训练周期全部损失 += 每名患者训练损失.detach().item()
            # 当前训练周期全部损失 += 训练损失.detach().item()
            # 训练循环.set_description(f'训练周期 [{当前训练周期}/{命令行参数.max_epoch}]')  # 设置进度条标题
            # 训练循环.set_postfix(训练损失 = 每名患者训练损失.detach().item())  # 每一批训练都更新损失


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


if __name__ == '__main__':
    网络模型 = model.融合网络无cls()
    网络模型.to(硬件设备)
    优化器 = torch.optim.Adam(网络模型.parameters())
    训练模型(网络模型, 优化器, 硬件设备, 无标签训练命令行参数)