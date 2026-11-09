# -*- coding: utf-8 -*-
"""kociemba 求解器的对外接口。

优先使用 pip 安装的官方 kociemba 包（原生 C 实现）；
本项目因缺少 MSVC 无法编译 C 扩展，故内置了官方 1.2.1 的
纯 Python 后端 pykociemba 作为回退，两者接口一致。
"""

from __future__ import annotations

# pykociemba 后端（已平铺到本目录）：纯 Python 的 Kociemba 两阶段算法
from .search import Search

# 官方约定的错误码 -> 中文提示
_ERRORS = {
    "Error 1": "6 种颜色各 9 个的约束不满足（某种颜色数量不对）",
    "Error 2": "12 条棱块没有恰好各出现一次",
    "Error 3": "翻棱错误：有一条棱必须翻转",
    "Error 4": "8 个角块没有恰好各出现一次",
    "Error 5": "扭角错误：有一个角必须扭转",
    "Error 6": "奇偶错误：需要交换两个角块或两条棱块",
    "Error 7": "给定深度内无解",
    "Error 8": "超时，限定时间内无解",
}


def solve(cubestring: str, max_depth: int = 24) -> str:
    """求解魔方：输入 54 字符 URFDLB 状态串，返回空格分隔的复原公式。"""
    # solution(状态串, 最大步数, 时间上限秒, 是否只找一个解)
    res = Search().solution(cubestring, max_depth, 1000, False).strip()
    if res in _ERRORS:  # 官方约定：出错时返回 "Error N" 字符串而非抛异常
        raise ValueError(_ERRORS[res])
    return res


__all__ = ["solve"]
