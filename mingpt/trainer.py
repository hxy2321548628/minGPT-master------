import time
from collections import defaultdict
import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN


class Trainer:

    @staticmethod
    def get_default_config():
        """
        获取默认配置，返回一个包含训练超参数的配置对象。
        """
        C = CN()
        # 设备选择，'auto'表示自动选择设备（CUDA或者CPU）
        C.device = "auto"
        # dataloader 参数
        C.num_workers = 4  # 加载数据时的工作线程数
        # 优化器参数
        C.max_iters = None  # 最大训练迭代次数
        C.batch_size = 64  # 每个批次的样本数量
        C.learning_rate = 3e-4  # 学习率
        C.betas = (0.9, 0.95)  # Adam优化器的beta参数
        C.weight_decay = 0.1  # 权重衰减（仅适用于矩阵乘法权重）
        C.grad_norm_clip = 1.0  # 梯度裁剪的最大值
        return C

    def __init__(self, config, model, train_dataset):
        """
        初始化训练器。

        参数:
            config (CfgNode): 配置对象，包含训练参数。
            model (nn.Module): 需要训练的模型。
            train_dataset (Dataset): 训练数据集。
        """
        self.config = config
        self.model = model
        self.optimizer = None
        self.train_dataset = train_dataset
        self.callbacks = defaultdict(list)

        # 选择训练设备（CPU或CUDA）
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("Running on device", self.device)

        # 训练过程中需要的其他变量
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        """
        为指定事件添加回调函数。
        """
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        """
        设置指定事件的回调函数，替换原有回调。
        """
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        """
        触发指定事件的所有回调函数。
        """
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self):
        """
        训练模型的核心方法。该方法会执行训练过程，包括前向传播、反向传播以及优化更新等步骤。
        """
        model, config = self.model, self.config

        # 设置优化器
        self.optimizer = model.configure_optimizers(config)

        # 设置数据加载器
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(
                self.train_dataset, replacement=True, num_samples=int(1e10)
            ),
            shuffle=False,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        # 设置模型为训练模式
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()  # 记录当前时间
        data_iter = iter(train_loader)

        # 无限循环，直到达到最大迭代次数或者训练结束
        while True:
            try:
                # 获取下一个批次的数据
                batch = next(data_iter)
            except StopIteration:
                # 如果数据迭代完，重新初始化迭代器
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # 将数据移到指定的设备（CPU/GPU）
            batch = [t.to(self.device) for t in batch]
            x, y = batch  # x是输入，y是标签

            # 前向传播
            logits, self.loss = model(x, y)

            # 反向传播，更新参数
            model.zero_grad(set_to_none=True)  # 清除梯度
            self.loss.backward()  # 反向传播
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_norm_clip
            )  # 梯度裁剪
            self.optimizer.step()  # 更新参数

            # 触发批次结束时的回调
            self.trigger_callbacks("on_batch_end")

            # 更新训练迭代信息
            self.iter_num += 1
            tnow = time.time()
            self.iter_dt = tnow - self.iter_time
            self.iter_time = tnow

            # 如果达到了最大迭代次数，结束训练
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
