'''
import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ monotonous bookkeeping """
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)
'''

import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------


def set_seed(seed):
    """
    设置随机种子，确保结果的可复现性。

    参数：
    seed (int): 随机种子
    """
    random.seed(seed)  # 设置 Python 的随机数生成器种子
    np.random.seed(seed)  # 设置 Numpy 的随机数生成器种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机数生成器种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 的随机数生成器种子


def setup_logging(config):
    """
    设置日志记录，主要用于记录训练参数、配置和命令行参数。

    参数：
    config (CfgNode): 配置信息
    """
    work_dir = config.system.work_dir  # 获取工作目录
    # 如果工作目录不存在，则创建它
    os.makedirs(work_dir, exist_ok=True)

    # 将命令行参数保存到 args.txt 文件
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))

    # 将配置保存到 config.json 文件
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))


class CfgNode:
    """
    配置节点类，灵感来自于 yacs，主要用于存储和管理配置。
    支持配置的嵌套和更新。
    """

    def __init__(self, **kwargs):
        """
        初始化配置节点。

        参数：
        kwargs: 配置的键值对
        """
        self.__dict__.update(kwargs)  # 更新当前实例的属性

    def __str__(self):
        """打印配置节点信息"""
        return self._str_helper(0)

    def _str_helper(self, indent):
        """
        递归地将配置以缩进格式打印出来，适应嵌套结构。

        参数：
        indent (int): 当前缩进层级
        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append(f"{k}:\n")
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f"{k}: {v}\n")
        parts = [" " * (indent * 4) + p for p in parts]  # 根据层级调整缩进
        return "".join(parts)

    def to_dict(self):
        """
        将配置节点转换为字典格式。
        """
        return {
            k: v.to_dict() if isinstance(v, CfgNode) else v
            for k, v in self.__dict__.items()
        }

    def merge_from_dict(self, d):
        """
        从字典更新配置项。

        参数：
        d (dict): 要合并的字典
        """
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        从命令行参数中更新配置。
        命令行参数格式为 `--arg=value`，支持通过 `.` 来表示嵌套属性。

        参数：
        args (list): 命令行参数
        """
        for arg in args:
            keyval = arg.split("=")
            assert len(keyval) == 2, f"每个参数应该是 --arg=value 格式，得到的是 {arg}"
            key, val = keyval  # 拆分参数名和参数值

            # 尝试将值转换为 Python 对象
            try:
                val = literal_eval(
                    val
                )  # 如果是合法的 Python 表达式（如列表、字典等），会被转换
            except ValueError:
                pass  # 如果转换失败，则保持字符串原样

            # 查找并更新属性
            assert key[:2] == "--"  # 确保以 '--' 开头
            key = key[2:]  # 移除 '--'
            keys = key.split(".")  # 根据 '.' 分割成多个层级
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)  # 逐层查找
            leaf_key = keys[-1]

            # 确保目标属性存在
            assert hasattr(obj, leaf_key), f"{key} 不是配置中的有效属性"

            # 更新该属性
            print(f"命令行参数覆盖配置项 {key}，新值为 {val}")
            setattr(obj, leaf_key, val)
