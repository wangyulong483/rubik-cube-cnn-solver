#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""魔方识别：CNN 判色 + 六面扫描求解（固定机位 / 聚光灯 / 纯色背景）

  1. config.json 里存着画面中魔方面的四个角点，透视变换把这一面压平成 450×450；
  2. 等分 3×3，每格取中心 60×60 缩成 32×32，九张小图一起喂 CNN，argmax 就是颜色；
  3. 九格成一面的字母，按 U-R-F-D-L-B 采完六面交给 kociemba 求解；
  4. 窗口下半是六面展开图（U 在上，L F R B 一行，D 在下）。相邻两面的边颜色对得上，
     才说明六面朝向采对了，这一步错了解法必然拧不回原样。
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

import kociemba
import ui

#常量

CAMERA_INDEX = 1             # 带聚光灯的外接 USB 摄像头；换机位就改这一行
CAMERA_SIZE = (1280, 720)    # 请求的采集分辨率；角点是归一化坐标，分辨率变了也不会错位

WARP = 450                   # 压平后的正方形边长
CELL = WARP // 3             # 每格 150
SAMPLE = 60                  # 每格只取中心 60×60 采样
PATCH = 32                   # 采样区缩成 32×32 喂 CNN

SCAN_ORDER = "URFDLB"        # 采集顺序，和 kociemba 的面序一致

CONFIG_PATH = Path(__file__).with_name("config.json")
MODEL_PATH = Path(__file__).with_name("cnn.pt")
SOLUTION_PATH = Path(__file__).with_name("solution.txt")

#模型

class CubeNet(nn.Module):
    """32×32 贴纸 → 6 个颜色的打分。两层卷积，样本几百张就够训。"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 8 * 8, 64), nn.ReLU(), nn.Linear(64, len(ui.COLORS)),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load_model():
    model = CubeNet()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model


# 角点文件

def default_config():
    """没有 config.json 时的角点：左上、右上、右下、左下，用画面宽高的比例。"""
    return {"quad": [[0.392, 0.311], [0.681, 0.309], [0.681, 0.813], [0.392, 0.815]]}


def load_config():
    if not CONFIG_PATH.exists():
        return default_config()
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


#识别

def warp_face(frame, quad):
    """四个归一化角点 → 450×450 的正方形，顺手把透视畸变拉直。"""
    h, w = frame.shape[:2]
    src = np.float32([[x * w, y * h] for x, y in quad])
    dst = np.float32([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]])
    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (WARP, WARP))


def cell_boxes():
    """9 个格子中心 60×60 采样区的左上角坐标，顺序逐行从左到右。"""
    offset = (CELL - SAMPLE) // 2
    return [(c * CELL + offset, r * CELL + offset) for r in range(3) for c in range(3)]


def cells(warp):
    """9 个格子的 32×32 BGR 小图。"""
    return [cv2.resize(warp[y:y + SAMPLE, x:x + SAMPLE], (PATCH, PATCH),
                       interpolation=cv2.INTER_AREA) for x, y in cell_boxes()]


def to_batch(patches):
    """一批 32×32 BGR 小图 → (N, 3, 32, 32) 张量：转 RGB，归一化到 [-2, 2]。"""
    array = np.stack(patches).astype(np.float32) / 255.0
    rgb = array[..., ::-1].copy().transpose(0, 3, 1, 2)
    return torch.from_numpy((rgb - 0.5) / 0.25)


def classify(model, warp):
    """这一面的 9 个格子 → 9 个颜色字母。"""
    with torch.no_grad():
        scores = model(to_batch(cells(warp)))
    return [ui.COLORS[i] for i in scores.argmax(1).tolist()]


def read_face(frame, cfg, model):
    """一帧画面 → (9 个颜色字母, 450×450 压平图)。"""
    warp = warp_face(frame, cfg["quad"])
    return classify(model, warp), warp


#求解

def turns_of(token):
    """"R" → 1，"R2" → 2，"R'" → 3。"""
    return 2 if token.endswith("2") else 3 if token.endswith("'") else 1


def parse_moves(solution):
    """把 "R U2 F' D" 拆成 [(轴序号, 转几次)]。"""
    return [(SCAN_ORDER.index(token[0]), turns_of(token)) for token in solution.split()]


def solution_steps(solution, faces):
    """解法 → [(第几步, 面字母, 这面的中心色, 转几次)]，给画面和 solution.txt 用。"""
    return [(i, token[0], faces[token[0]][4], turns_of(token))
            for i, token in enumerate(solution.split(), 1)]


def apply_solution(state, solution):
    """把解法逐步乘回状态串，用来验证解法真的能复原。"""
    from kociemba.cubiecube import moveCube
    from kociemba.facecube import FaceCube

    cube = FaceCube(state).toCubieCube()
    for axis, turns in parse_moves(solution):
        for _ in range(turns):
            cube.multiply(moveCube[axis])
    return cube.toFaceCube().to_String()


def is_solved(state):
    return all(state[i * 9:(i + 1) * 9] == state[i * 9] * 9 for i in range(6))


def solve_faces(faces):
    """六个面的 9 个颜色 → 54 字符状态串 → kociemba 解法，并写 solution.txt。"""
    centers = [faces[face][4] for face in SCAN_ORDER]                # 中心块颜色决定这一面是哪个字母
    if len(set(centers)) != 6:
        raise ValueError("中心块颜色重复：" + "".join(ui.COLOR_CN[c] for c in centers))

    center_to_face = dict(zip(centers, SCAN_ORDER))
    state = "".join(center_to_face[color] for face in SCAN_ORDER for color in faces[face])

    wrong = [f"{ui.COLOR_CN[color]} {sum(face.count(color) for face in faces.values())} 格"
             for color in ui.COLORS if sum(face.count(color) for face in faces.values()) != 9]
    if wrong:
        raise ValueError("颜色数不对：" + "、".join(wrong))

    start = time.time()
    solution = kociemba.solve(state)
    verified = is_solved(apply_solution(state, solution))
    steps = [ui.step_text(*item) for item in solution_steps(solution, faces)]
    SOLUTION_PATH.write_text(
        "\n".join([
            f"state: {state}",
            f"solution: {solution}",
            f"moves: {len(steps)}",
            f"time: {datetime.now():%Y-%m-%d %H:%M:%S}",
            f"verified: {'yes' if verified else 'no'}",
            "",
            "每一步都是：把括号里那一面正对自己，再按顺/逆时针拧；颜色是扫面时识别出的中心色",
            *steps,
        ]) + "\n",
        encoding="utf-8",
    )
    return solution, verified, time.time() - start


#状态提示

def status_lines(scanning, faces, error, solution=""):
    """底部展开图旁边写什么，取决于现在处于哪一步。"""
    done = len(faces)
    if solution:
        return ["解法已生成", "空格 = 翻页    S = 重新扫描    ESC = 退出",
                "拧的时候整块魔方别转，只拧要转的那一个面",
                "展开图：相邻两面的边颜色应当对得上"]
    if error:
        return ["求解失败", error[:24], "R = 重拍上一面    S = 退出扫描",
                "展开图：相邻两面的边颜色应当对得上"]
    if scanning and done < 6:
        face = SCAN_ORDER[done]
        return [f"扫描 {done + 1}/6    请展示 {face} 面（{ui.FACE_CN[face]}）", ui.FACE_HINT[face],
                "基准 白上绿前：右=红 左=橙 前=绿 后=蓝",
                "空格 = 采集    R = 重拍上一面    S = 退出扫描",
                "展开图：相邻两面的边颜色应当对得上"]
    if scanning:
        return ["六面已齐", "R = 重拍上一面    S = 退出扫描",
                "展开图：相邻两面的边颜色应当对得上"]
    return ["识别结果", "S = 六面扫描    ESC = 退出"]


#主循环

def main():
    if not MODEL_PATH.exists():
        raise SystemExit("还没有 cnn.pt：先跑 python train.py 标注格子训练一次")
    cfg = load_config()
    model = load_model()

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])
    cv2.namedWindow(ui.WINDOW, cv2.WINDOW_AUTOSIZE)

    scanning = False     # 是否在扫六面
    faces = {}           # 面字母 -> 9 个颜色
    solution = ""        # 最近一次的解法
    verified = False
    error = ""
    page = 0             # 解法翻到第几页

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        letters, warp = read_face(frame, cfg, model)
        view = ui.draw_camera(frame, cfg["quad"])
        pending = (SCAN_ORDER[len(faces)], letters) if scanning and len(faces) < 6 else None

        canvas = np.full((ui.VIEW_H + ui.BOTTOM_H, view.shape[1] + ui.FACE, 3), ui.BG, np.uint8)
        canvas[:ui.VIEW_H, :view.shape[1]] = ui.draw_solution(solution_steps(solution, faces), verified,
                                                              page, view.shape[1]) if solution else view
        canvas[:ui.VIEW_H, view.shape[1]:] = ui.draw_face(warp, letters)
        canvas[ui.VIEW_H + 12:ui.VIEW_H + 12 + ui.NET_H, 14:14 + ui.NET_W] = ui.draw_net(faces, pending)
        canvas = ui.overlay_text(canvas, status_lines(scanning, faces, error, solution),
                                 origin=(14 + ui.NET_W + 24, ui.VIEW_H + 16))
        cv2.imshow(ui.WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:                                   # ESC
            break
        elif key in (ord("s"), ord("S")):               # 进入 / 退出六面扫描
            scanning = not scanning
            faces.clear()
            solution, verified, error, page = "", False, "", 0
        elif key == ord(" "):
            if solution:                                # 有解法：翻页
                pages = max(1, -(-len(solution_steps(solution, faces)) // ui.STEPS_PER_PAGE))
                page = (page + 1) % pages
            elif scanning and len(faces) < 6:           # 扫描中：采集当前面
                faces[SCAN_ORDER[len(faces)]] = letters
                if len(faces) == 6:
                    try:
                        solution, verified, _ = solve_faces(faces)
                        page = 0
                    except Exception as exc:            # 采错面时 kociemba 会拒收，提示重拍
                        error = str(exc)
        elif key in (ord("r"), ord("R")) and scanning and faces:
            faces.popitem()                             # 撤销最近采集的一面
            solution, verified, error, page = "", False, "", 0

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
