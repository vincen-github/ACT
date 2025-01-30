from .act import ACT
from .haochen22 import Haochen22
from .barlow_twins import BarlowTwins


METHOD_LIST = ["act", "haochen22", "barlow_twins"]


def get_method(name):
    assert name in METHOD_LIST
    if name == "act":
        return ACT
    elif name == "haochen22":
        return Haochen22
    elif name == "barlow_twins":
        return BarlowTwins