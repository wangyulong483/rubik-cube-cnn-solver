#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------- 尺寸与配色

VIEW_H = 360                 # 所有面板统一高度
FACE = 360                   # 压平后一面 / 统计栏的边长
STATS_W = 300                # train.py 右侧统计栏宽度

NET_CELL = 18                # 展开图每格边长
NET_GAP = 8                  # 展开图相邻两面的间距
NET_W = 4 * 3 * NET_CELL + 3 * NET_GAP     # 展开图宽度：一行摆 L F R B
NET_H = 3 * 3 * NET_CELL + 2 * NET_GAP     # 展开图高度：U 上、中间一行、D 下
BOTTOM_H = NET_H + 24        # 窗口底部留给展开图的高度
NET_POS = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}

BG = (26, 26, 26)            # 空白处底色（BGR）
FONT_FILE = "C:/Windows/Fonts/msyh.ttc"

COLORS = ("W", "Y", "R", "O", "B", "G")
COLOR_CN = {"W": "白", "Y": "黄", "R": "红", "O": "橙", "B": "蓝", "G": "绿"}
COLOR_RGB = {"W": (245, 245, 245), "Y": (235, 205, 60), "R": (215, 60, 55),
             "O": (240, 140, 40), "B": (55, 105, 210), "G": (60, 175, 80)}
COLOR_BGR = {"W": (240, 240, 240), "Y": (60, 215, 235), "R": (60, 60, 215),
             "O": (60, 145, 245), "B": (215, 105, 55), "G": (75, 175, 75)}
FACE_CN = {"U": "上·白", "R": "右·红", "F": "前·绿", "D": "下·黄", "L": "左·橙", "B": "后·蓝"}
FACE_BGR = {"U": (240, 240, 240), "D": (70, 220, 240), "F": (80, 180, 70),
            "B": (200, 110, 60), "R": (70, 70, 215), "L": (50, 155, 245)}

# 扫描提示：每一面在镜头前怎么摆（决定了 3×3 的行列方向）
FACE_HINT = {
    "U": "白心朝镜头，绿心朝下",
    "R": "红心朝镜头，白心朝上",
    "F": "绿心朝镜头，白心朝上",
    "D": "黄心朝镜头，绿心朝上",
    "L": "橙心朝镜头，白心朝上",
    "B": "蓝心朝镜头，白心朝上",
}

MOVE_CN = {"U": "顶面", "D": "底面", "F": "前面", "B": "后面", "L": "左面", "R": "右面"}
MOVE_POS = {"U": "上", "D": "下", "F": "前", "B": "后", "L": "左", "R": "右"}
TURN_CN = {1: "顺时针 90°", 2: "转 180°", 3: "逆时针 90°"}
STEPS_PER_PAGE = 22          # 一页 22 步：两列，每列 11 行

WINDOW = "魔方识别"
TRAIN_WINDOW = "标注训练"

# ---------------------------------------------------------------- 小工具

_fonts = {}


def font(size):
    """中文字体按字号缓存，别每帧都读一次 ttc。"""
    if size not in _fonts:
        _fonts[size] = ImageFont.truetype(FONT_FILE, size)
    return _fonts[size]


def grid(image):
    """在方形图上画 3×3 绿格线。"""
    side = image.shape[0]
    for t in (1 / 3, 2 / 3):
        p = int(side * t)
        cv2.line(image, (p, 0), (p, side), (0, 255, 0), 1)
        cv2.line(image, (0, p), (side, p), (0, 255, 0), 1)


def put_letter(image, letter, index, side):
    """第 index 格中心写一个黑描边绿字。"""
    step = side // 3
    x, y = (index % 3) * step + step // 2, (index // 3) * step + step // 2
    cv2.putText(image, letter, (x - 14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 7)
    cv2.putText(image, letter, (x - 14, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2)


# ---------------------------------------------------------------- main.py 用

def draw_camera(frame, quad, height=VIEW_H):
    """左侧相机画面：缩放到固定高度，画出四角和 3×3 网格。"""
    h, w = frame.shape[:2]
    view = cv2.resize(frame, (int(w * height / h), height))
    pts = np.float32([[x * view.shape[1], y * height] for x, y in quad])
    cv2.polylines(view, [pts.astype(np.int32)], True, (0, 255, 0), 2)
    for t in (1 / 3, 2 / 3):            # 网格线：对边按比例插值连起来
        top, bottom = pts[0] * (1 - t) + pts[1] * t, pts[3] * (1 - t) + pts[2] * t
        left, right = pts[0] * (1 - t) + pts[3] * t, pts[1] * (1 - t) + pts[2] * t
        cv2.line(view, tuple(top.astype(int)), tuple(bottom.astype(int)), (0, 255, 0), 1)
        cv2.line(view, tuple(left.astype(int)), tuple(right.astype(int)), (0, 255, 0), 1)
    return view


def draw_face(warp, letters):
    """右侧面板：压平后的这一面，格子上写识别结果。"""
    face = cv2.resize(warp, (FACE, FACE))
    grid(face)
    for i, letter in enumerate(letters):
        put_letter(face, letter, i, FACE)
    return face


def step_text(index, face, color, turns):
    """一句人话：3. 红面(右) 正对自己，顺时针 90°。颜色取这一面识别出的中心色。"""
    return f"{index}. {COLOR_CN[color]}面({MOVE_POS[face]}) 正对自己，{TURN_CN[turns]}"


def step_icon(color, size=26):
    """一小张图：这一步拧的是哪一面，填的就是识别出的那个颜色。方向看文字。"""
    icon = np.full((size, size, 3), COLOR_BGR[color], np.uint8)
    for t in (1 / 3, 2 / 3):
        cell = int(size * t)
        cv2.line(icon, (cell, 0), (cell, size), (40, 40, 40), 1)
        cv2.line(icon, (0, cell), (size, cell), (40, 40, 40), 1)
    return icon


def draw_solution(steps, verified, page, width):
    """左侧解法页：一步一行配一张小图；steps 是 [(第几步, 面字母, 转几次)]。"""
    image = Image.new("RGB", (width, VIEW_H), BG[::-1])
    draw = ImageDraw.Draw(image)
    pages = max(1, -(-len(steps) // STEPS_PER_PAGE))

    draw.text((14, 8), f"解法 {len(steps)} 步" +
              ("（已验证能复原）" if verified else "（验证未通过，建议重拍）"),
              font=font(18), fill=(240, 240, 240))
    draw.text((14, 34), "每句 = 把那一面正对自己拧；括号里是它在魔方上的位置",
              font=font(13), fill=(150, 205, 150))
    if pages > 1:
        draw.text((width - 150, 10), f"第 {page + 1}/{pages} 页，空格翻页",
                  font=font(13), fill=(200, 200, 120))

    column_width = (width - 24) // 2
    for slot, (index, face, color, turns) in enumerate(steps[page * STEPS_PER_PAGE:(page + 1) * STEPS_PER_PAGE]):
        column, row = divmod(slot, STEPS_PER_PAGE // 2)
        x, y = 14 + column * column_width, 58 + row * 26
        icon = cv2.cvtColor(step_icon(color), cv2.COLOR_BGR2RGB)
        image.paste(Image.fromarray(icon), (x, y))
        draw.text((x + 30, y + 4), step_text(index, face, color, turns),
                  font=font(14), fill=(225, 225, 225))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def overlay_text(canvas, lines, origin=(14, 14)):
    """把中文状态文字描边压到画面上（cv2 写不了中文，借一下 Pillow）。"""
    img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    x, y = origin
    for line in lines:
        draw.text((x, y), line, font=font(19), fill=(240, 240, 240),
                  stroke_width=3, stroke_fill=(20, 20, 20))
        y += 28
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_net(faces, pending=None):
    """底部展开图：U 在上、LF RB 一行、D 在下，格子里是识别出来的颜色。

    pending = (面字母, 9 个颜色) 时，把"镜头里正在看、还没按空格确认"的这一面
    用黄框画在它将来要占的位置上，方便按格跟真魔方对着比。
    """
    net = np.full((NET_H, NET_W, 3), BG, np.uint8)
    for face, (column, row) in NET_POS.items():
        x, y = column * (3 * NET_CELL + NET_GAP), row * (3 * NET_CELL + NET_GAP)
        letters = faces.get(face)
        if letters is None and pending and pending[0] == face:
            letters, frame_color = pending[1], (0, 220, 255)
        else:
            frame_color = (0, 255, 0) if letters else (60, 60, 60)
        for i in range(9):
            cx, cy = x + (i % 3) * NET_CELL, y + (i // 3) * NET_CELL
            fill = COLOR_BGR[letters[i]] if letters else (38, 38, 38)
            cv2.rectangle(net, (cx, cy), (cx + NET_CELL - 1, cy + NET_CELL - 1), fill, -1)
            cv2.rectangle(net, (cx, cy), (cx + NET_CELL - 1, cy + NET_CELL - 1), (20, 20, 20), 1)
        cv2.rectangle(net, (x - 2, y - 2), (x + 3 * NET_CELL + 1, y + 3 * NET_CELL + 1),
                      frame_color, 2 if pending and pending[0] == face else 1)
        cv2.putText(net, face, (x + 2, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (110, 110, 110), 1)
    return net


# ---------------------------------------------------------------- train.py 用

def draw_warp(warp, labels, cell):
    """中间：压平后的一面，黄框是当前要标的格子，字母是已经贴上的标签。"""
    face = cv2.resize(warp, (FACE, FACE))
    grid(face)
    step = FACE // 3
    row, col = divmod(cell, 3)
    cv2.rectangle(face, (col * step + 4, row * step + 4),
                  (col * step + step - 4, row * step + step - 4), (0, 220, 255), 3)
    for i, label in enumerate(labels):
        if label is not None:
            put_letter(face, label, i, FACE)
    return face


def draw_stats(samples, cell, accuracy):
    """右侧统计：每种颜色攒了多少个格子，以及训练结果和按键提示。"""
    panel = Image.fromarray(np.full((VIEW_H, STATS_W, 3), BG[::-1], np.uint8))
    draw = ImageDraw.Draw(panel)
    draw.text((14, 10), "样本统计", font=font(20), fill=(240, 240, 240))
    y = 46
    for i, name in enumerate(COLORS):
        count = sum(1 for _, label in samples if label == i)
        draw.text((14, y), f"{name} {COLOR_CN[name]}", font=font(18), fill=(200, 200, 200))
        draw.rectangle([86, y + 8, 86 + min(count, 80) * 2, y + 18], fill=COLOR_RGB[name])
        draw.text((250, y), f"{count}", font=font(18), fill=(200, 200, 200))
        y += 34

    y += 6
    draw.text((14, y), f"当前格 {cell + 1}", font=font(17), fill=(150, 220, 150))
    y += 26
    if accuracy is not None:
        draw.text((14, y), f"验证准确率 {accuracy:.1%}", font=font(17), fill=(120, 230, 120))
        y += 28
    for line in ("w y r o b g 贴标签    1~9 选格子", "T 训练保存 cnn.pt    ESC 退出"):
        draw.text((14, y), line, font=font(14), fill=(170, 170, 170))
        y += 22
    return np.array(panel)[:, :, ::-1].copy()
