#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""标注 + 训练：给 3×3 的九个格子按颜色贴标签，训出 main.py 用的 cnn.pt。

  1. 相机固定不动，config.json 里的四个角点就是屏幕上的绿框；把魔方摆进框里、让一面填满
     就算对准，不用在画面上拖角点；
  2. 中间 9 格从第 1 格开始自动选中，按 w/y/r/o/b/g 给它贴标签，贴完自动跳到下一格，
     照 1→9 的顺序打完这一面；九格打完自动清空，转个面接着打；
  3. 每种颜色攒到三四十个格子后按 T 训练。屏幕上报验证准确率，权重写进 cnn.pt，
     main.py 下次启动读它，不用改任何阈值。
"""
import random

import cv2
import numpy as np
import torch

import ui
from main import (CAMERA_INDEX, CAMERA_SIZE, MODEL_PATH, CubeNet, cells,
                  load_config, to_batch, warp_face)

KEYS = {ord("w"): 0, ord("y"): 1, ord("r"): 2, ord("o"): 3, ord("b"): 4, ord("g"): 5}
DIGITS = {ord(str(i + 1)): i for i in range(9)}

state = {"quad": None, "cell": 0, "labels": [None] * 9, "samples": [],
         "accuracy": None}


# ---------------------------------------------------------------- 标注

def label_cell(color):
    """把当前格贴上颜色标签，顺便存一份样本，然后跳到下一格。"""
    warp = warp_face(state["frame"], state["quad"])
    patch = cells(warp)[state["cell"]]
    state["labels"][state["cell"]] = ui.COLORS[color]
    state["samples"].append((patch, color))
    state["cell"] += 1
    if state["cell"] == 9:            # 这一面标完了，清空等下一个面
        state["cell"] = 0
        state["labels"] = [None] * 9


# ---------------------------------------------------------------- 训练

def jitter(batch):
    """随机翻转 / 旋转 / 整体明暗，几百张样本也能扛一点光照变化。"""
    if random.random() < 0.5:
        batch = torch.flip(batch, [3])
    if random.random() < 0.5:
        batch = torch.flip(batch, [2])
    batch = torch.rot90(batch, random.randrange(4), [2, 3])
    return batch * (0.8 + 0.4 * random.random()) + (random.random() - 0.5) * 0.4


def accuracy(model, patches, labels):
    if not patches:
        return 0.0
    with torch.no_grad():
        guess = model(to_batch(patches)).argmax(1).tolist()
    return sum(int(g == l) for g, l in zip(guess, labels)) / len(labels)


def train(samples):
    """样本 → CubeNet，返回验证准确率；每类留五分之一做验证。"""
    train_x, train_y, val_x, val_y = [], [], [], []
    for color in range(len(ui.COLORS)):
        group = [patch for patch, label in samples if label == color]
        cut = len(group) // 5
        val_x += group[:cut]
        val_y += [color] * cut
        train_x += group[cut:]
        train_y += [color] * (len(group) - cut)

    model = CubeNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_all, y_all = to_batch(train_x), torch.tensor(train_y)
    best_score, best_weights = -1.0, None

    for epoch in range(40):
        order = torch.randperm(len(y_all))
        for i in range(0, len(order), 32):
            index = order[i:i + 32]
            loss = loss_fn(model(jitter(x_all[index])), y_all[index])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = accuracy(model, val_x or train_x, val_y or train_y)   # 样本太少没有验证集时看训练集
        if score >= best_score:
            best_score = score
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 9:
            print(f"epoch {epoch + 1:>2}  loss {loss.item():.3f}  验证 {score:.1%}")

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), MODEL_PATH)
    return best_score


# ---------------------------------------------------------------- 主循环

def main():
    state["quad"] = load_config()["quad"]

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_SIZE[1])

    cv2.namedWindow(ui.TRAIN_WINDOW, cv2.WINDOW_AUTOSIZE)

    while True:
        ok, frame = camera.read()
        if not ok:
            break
        state["frame"] = frame

        view = ui.draw_camera(frame, state["quad"])
        canvas = np.full((ui.VIEW_H, view.shape[1] + ui.FACE + ui.STATS_W, 3), ui.BG, np.uint8)
        canvas[:, :view.shape[1]] = view
        canvas[:, view.shape[1]:view.shape[1] + ui.FACE] = ui.draw_warp(
            warp_face(frame, state["quad"]), state["labels"], state["cell"])
        canvas[:, view.shape[1] + ui.FACE:] = ui.draw_stats(
            state["samples"], state["cell"], state["accuracy"])
        cv2.imshow(ui.TRAIN_WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key in KEYS:
            label_cell(KEYS[key])
        elif key in DIGITS:
            state["cell"] = DIGITS[key]
        elif key in (ord("t"), ord("T")) and state["samples"]:
            print(f"训练 {len(state['samples'])} 个样本 …")
            state["accuracy"] = train(state["samples"])
            print(f"验证准确率 {state['accuracy']:.1%}，已写入 {MODEL_PATH.name}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
