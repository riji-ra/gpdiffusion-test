import numpy as np
from numba import njit, int64, int32
import cv2
import torchvision
from copy import deepcopy
from diffusers import DiffusionPipeline, AutoencoderTiny
from flask import Flask, jsonify, request
import gc
import torch
import glob
import time
from numba import jit
import tqdm
import hashlib
import os
import json

PE = lambda a: np.meshgrid(np.linspace(0, 1, a.shape[1]), np.linspace(0, 1, a.shape[0]))

TT = lambda a: np.concatenate((np.ones(1), (a[:-2] + a[1:-1]*2 + a[2:]) * 0.25, np.ones(1)))
TT2 = lambda a: np.concatenate((np.ones(1), ((a[:-2] - a[1:-1]) ** 2 + (a[1:-1] - a[2:]) ** 2) * 0.5, np.ones(1)))

i0__ = {
    "A": lambda a: a ** 2,
    "B": lambda a: np.abs(a),
    "C": lambda a: np.sin(a * np.pi),
    "D": lambda a: np.concatenate((a[-1:], a[:-1]), axis=0),
    "E": lambda a: np.concatenate((a[1:], a[:1]), axis=0),
    "F": lambda a: np.concatenate((a[:, -1:], a[:, :-1]), axis=1),
    "G": lambda a: np.concatenate((a[:, 1:], a[:, :1]), axis=1),
    "H": lambda a: np.fliplr(a),
    "I": lambda a: np.flipud(a),
    "J": lambda a: a * 2,
    "K": lambda a: a * 10,
    "L": lambda a: a * 0.9,
    "M": lambda a: a * 0.1,
    "N": lambda a: a + 1,
    "O": lambda a: -a,
    "P": lambda a: cv2.GaussianBlur(a, (3, 3), 0),
    "Q": lambda a: cv2.GaussianBlur(a, (7, 7), 0),
    "R": lambda a: a * 0 + np.std(a),
    "S": lambda a: a * 0 + np.mean(a),
    "T": lambda a: np.exp(- (PE(a)[0]**2 + PE(a)[1]**2) / (np.var(a) + 0.01)),
    "U": lambda a: a * 0 + np.max(a),
    "V": lambda a: a * 0 + np.min(a),
    "W": lambda a: (a - np.mean(a, axis=0, keepdims=True)) / (np.std(a, axis=0, keepdims=True) + 0.01),
    "X": lambda a: (a - np.mean(a)) / (np.std(a) + 0.01),
    "Y": lambda a: a * 0 + np.mean(a, axis=0, keepdims=True),
    "Z": lambda a: a * 0 + np.std(a, axis=0, keepdims=True),
    "AA": lambda a: np.concatenate((a[::2], a[1::2]), axis=0),
    "AB": lambda a: np.concatenate((a[:, ::2], a[:, 1::2]), axis=-1),
    "AC": lambda a: a * (np.tanh(a) + 1),
    "AD": lambda a: np.tanh(a),
    "AE": lambda a: np.fft.fft2(a).real / a.shape[0],
    "AF": lambda a: np.fft.fft2(a).imag / a.shape[0],
    "AG": lambda a: np.sort(a.flatten()).reshape(a.shape),
    "AH": lambda a: a[np.argsort(np.mean(a, axis=-1))],
    "AI": lambda a: a[np.argsort(np.std(a, axis=-1))],
    "AJ": lambda a: a[:, np.argsort(np.mean(a, axis=0))],
    "AK": lambda a: a[:, np.argsort(np.std(a, axis=0))],
    "AL": lambda a: cv2.resize(a, (a.shape[1]*2, a.shape[0]*2))[a.shape[0]//2:a.shape[0]*2-a.shape[0]+a.shape[0]//2, a.shape[1]//2:a.shape[1]*2-a.shape[1]+a.shape[1]//2],
    "AM": lambda a: np.flip(a),
    "AN": lambda a: cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1]/2, a.shape[0]/2), 90, 1), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AO": lambda a: np.abs(a) ** 0.5,
    "AP": lambda a: np.concatenate((a[-4:], a[:-4]), axis=0),
    "AQ": lambda a: np.concatenate((a[4:], a[:4]), axis=0),
    "AR": lambda a: np.concatenate((a[:, -4:], a[:, :-4]), axis=1),
    "AS": lambda a: np.concatenate((a[:, 4:], a[:, :4]), axis=1),
    "AT": lambda a: np.abs(a) ** (1/3) * np.sign(a),
    "AU": lambda a: a / (np.mean(a ** 2) + 0.01),
    "AV": lambda a: a / (np.mean(a ** 2) ** 0.5 + 0.01),
    "AW": lambda a: a - 1,
    "AX": lambda a: (a - np.mean(a, axis=1, keepdims=True)) / (np.std(a, axis=1, keepdims=True) + 0.01),
    "AY": lambda a: a + 0.1,
    "AZ": lambda a: a + 0.5,
    "BA": lambda a: a - 0.5,
    "BB": lambda a: a - 1,
    "BC": lambda a: 1 - a,
    "BD": lambda a: a / 2,
    "BE": lambda a: a - np.mean(a),
    "BF": lambda a: cv2.GaussianBlur(a, (9, 9), 0),
    "BG": lambda a: np.concatenate((a[-8:], a[:-8]), axis=0),
    "BH": lambda a: np.concatenate((a[8:], a[:8]), axis=0),
    "BI": lambda a: np.concatenate((a[:, -8:], a[:, :-8]), axis=1),
    "BJ": lambda a: np.concatenate((a[:, 8:], a[:, :8]), axis=1),
    "BH": lambda a: a * 0.5,
    "BI": lambda a: a * -0.5,
    #"BJ": lambda a: a.T,
    #"BK": lambda a: np.fliplr(a.T),
    #"BL": lambda a: np.flipud(a.T),
    "BM": lambda a: a ** 3,
    "BN": lambda a: a + 2.0,
    "BO": lambda a: a + 5.0,
    "BP": lambda a: a - 2.0,
    "BQ": lambda a: np.tanh(a),
    "BR": lambda a: a * np.log(np.square(a) + 1e-6),
    "BS": lambda a: np.concatenate((a[-2:], a[:-2]), axis=0),
    "BT": lambda a: np.concatenate((a[2:], a[:2]), axis=0),
    "BU": lambda a: np.concatenate((a[:, -2:], a[:, :-2]), axis=1),
    "BV": lambda a: np.concatenate((a[:, 2:], a[:, :2]), axis=1),
    "BW": lambda a: a * -2,
    "BX": lambda a: a * -4,
    "BY": lambda a: a * -0.1,
    "BZ": lambda a: a * -0.5,
    "CA": lambda a: ((np.concatenate((a[-1:], a[:-1]), axis=0) - a) ** 2 + (np.concatenate((a[1:], a[:1]), axis=0) - a) ** 2) ** 0.5,
    "CB": lambda a: ((np.concatenate((a[:, -1:], a[:, :-1]), axis=1) - a) ** 2 + (np.concatenate((a[:, 1:], a[:, :1]), axis=1) - a) ** 2) ** 0.5,
    "CC": lambda a: ((np.concatenate((a[:, -1:], a[:, :-1]), axis=1) - a) ** 2 + (np.concatenate((a[:, 1:], a[:, :1]), axis=1) - a) ** 2 + (np.concatenate((a[-1:], a[:-1]), axis=0) - a) ** 2 + (np.concatenate((a[1:], a[:1]), axis=0) - a) ** 2) ** 0.5,
    "CD": lambda a: (a - cv2.GaussianBlur(a, (9, 9), 0)) / np.sqrt(cv2.GaussianBlur(a**2 + 1e-6, (9, 9), 0)),
    "CF": lambda a: (a - cv2.GaussianBlur(a, (13, 13), 0)) / np.sqrt(cv2.GaussianBlur(a**2 + 1e-6, (13, 13), 0)),
}

i1_ = {
    "A": lambda a, b: a + b,
    "B": lambda a, b: a - b,
    "C": lambda a, b: a * b,
    "D": lambda a, b: a / (b**2 + 0.01),
    "E": lambda a, b: np.maximum(a, b),
    "F": lambda a, b: np.minimum(a, b),
    "G": lambda a, b: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b) / a.shape[0] ** 2).real,
    "H": lambda a, b: np.fft.ifft2(np.fft.fft2(np.tanh(a) + 1) * np.fft.fft2(np.tanh(b)) / a.shape[0] ** 2).real,
    #"I": lambda a, b: (a.T @ (np.tanh(b[:a.shape[0], :a.shape[0]]) + 1)).T / a.shape[0],
    #"J": lambda a, b: np.take(a.flatten(), np.asarray(np.floor(np.tanh(b.flatten()) * (b.flatten().shape[0] - 1)), dtype=np.int64)).reshape(a.shape),
    "K": lambda a, b: np.take(a.flatten(), np.argsort(b.flatten())).reshape(a.shape),
    "L": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (3, 3)) / 3**2),
    "M": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (5, 5)) / 5**2),
    "N": lambda a, b: np.fft.ifft(np.fft.fft(a.flatten()) * np.fft.fft(np.tanh(b).flatten()) / a.shape[0] ** 2).real.reshape(a.shape),
    "O": lambda a, b: np.sin(a * b * np.pi),
    #"P": lambda a, b: np.mean(np.sin((np.sin((np.repeat(np.stack(PE(a), axis=-1), a.shape[0]//2, axis=-1) @ a + np.mean(a, axis=0)[None, None]) / np.sqrt(a.shape[0]) * np.pi) @ b.T + np.mean(b, axis=1)[None, None]) / np.sqrt(a.shape[0]) * np.pi), axis=-1),
    "Q": lambda a, b: np.concatenate((a[::2], b[1::2]), axis=0),
    #"R": lambda a, b: np.take(np.mean(a, axis=1), np.asarray(np.floor(np.tanh(b.flatten()) * (b.shape[0] - 1)), dtype=np.int64)).reshape(a.shape),
    "T": lambda a, b: np.exp(- ((PE(a)[0]**2 - np.mean(a)) / (np.var(a) + 0.01) + (PE(a)[1]**2 - np.mean(b)) / (np.var(b) + 0.01))),
    "U": lambda a, b: a[np.argsort(np.mean(b, axis=-1))],
    "V": lambda a, b: a[:, np.argsort(np.mean(b, axis=0))],
    #"W": lambda a, b: (a.T @ (b[:a.shape[0], :a.shape[0]])).T / a.shape[0],
    "X": lambda a, b: (a - b) ** 2,
    "Y": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (11, 11)) / 11**2),
    "Z": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (25, 25)) / 25**2),
    "AA": lambda a, b: cv2.warpAffine(a, cv2.getRotationMatrix2D((a.shape[1]/2, a.shape[0]/2), np.mean(b) * 360, np.std(b)), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AB": lambda a, b: cv2.warpPerspective(a, cv2.resize(b, (3, 3)), (a.shape[1], a.shape[0]), borderMode=cv2.BORDER_REPLICATE),
    "AC": lambda a, b: np.concatenate((a[:a.shape[0]//2], b[-b.shape[0]//2:]), axis=0),
    "AD": lambda a, b: np.concatenate((a[:, :a.shape[1]//2], b[:, -b.shape[1]//2:]), axis=1),
    "AE": lambda a, b: np.fft.ifft2(a + b*1j).real * a.shape[0]**2,
    #"AF": lambda a, b: a[np.asarray(np.floor((b.shape[0] - 1) * (np.tanh(np.mean(b, axis=-1)) + 1) * 0.5), dtype=np.int64)],
    #"AG": lambda a, b: a[:, np.asarray(np.floor((b.shape[1] - 1) * (np.tanh(np.mean(b, axis=0)) + 1) * 0.5), dtype=np.int64)],
    "AH": lambda a, b: (a + b) / 2,
    "AI": lambda a, b: (a ** 2 + b ** 2) ** 0.5,
    "AJ": lambda a, b: np.concatenate((a[:, ::2], b[:, 1::2]), axis=-1),
    "AK": lambda a, b: a * (np.tanh(b) + 1),
    "AL": lambda a, b: b * (np.tanh(a) + 1),
    "AM": lambda a, b: a*2 - b,
    "AN": lambda a, b: a*0.75 + b*0.25,
    "AO": lambda a, b: a*0.333 + b*0.666,
    "AP": lambda a, b: a*0.25 + b*0.75,
    "AQ": lambda a, b: np.abs(a * b * b) ** 1/3 * np.sign(a * b * b),
    "AR": lambda a, b: np.abs(a * a * b) ** 1/3 * np.sign(a * a * b),
    "AS": lambda a, b: np.sin(a * b),
    "AT": lambda a, b: np.sin(a * np.mean(b) * np.pi),
    "AU": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (3, 3))))),
    "AV": lambda a, b: np.take(TT(np.take(a.flatten(), np.argsort(b.flatten()))), np.argsort(np.argsort(b.flatten()))).reshape(a.shape),
    "AW": lambda a, b: np.take(TT(np.take(b.flatten(), np.argsort(a.flatten()))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AX": lambda a, b: np.take(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AY": lambda a, b: np.take(TT2(np.take(b.flatten(), np.argsort(a.flatten()))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AZ": lambda a, b: np.take(TT(TT(TT(TT(TT(TT(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))))))))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BA": lambda a, b: np.take(TT(TT(TT(TT(np.take(b.flatten(), np.argsort(a.flatten())))))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BB": lambda a, b: np.take(TT2(TT2(np.take(b.flatten(), np.argsort(a.flatten())))), np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "BC": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (3, 3))) / 3**2),
    "BD": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (5, 5))) / 5**2),
    "BE": lambda a, b: cv2.filter2D(a, -1, np.tanh(cv2.resize(b, (7, 7))) / 7**2),
    "BF": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (3, 3))) + 1) / 3**2),
    "BG": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (5, 5))) + 1) / 5**2),
    "BH": lambda a, b: cv2.filter2D(a, -1, (np.tanh(cv2.resize(b, (7, 7))) + 1) / 7**2),
    "BI": lambda a, b: np.fft.ifft2(np.fft.fft2(a) ** 2 / np.fft.fft2(b)).real,
    "BJ": lambda a, b: np.fft.ifft2(np.fft.fft2(b) ** 2 / np.fft.fft2(a)).real,
    "BK": lambda a, b: np.fft.ifft2(np.fft.fft2(b) ** 2 / np.fft.fft2(np.tanh(a))).real,
    "BL": lambda a, b: cv2.filter2D(a, -1, cv2.resize(b, (15, 14)) / 15**2),
    "BM": lambda a, b: a - b*2,
    "BN": lambda a, b: a*3 - b*2,
    "BO": lambda a, b: a*4 - b*3,
}

i2_ = {
    "AA": lambda a, b, c: a + b + c,
    "AB": lambda a, b, c: a + b - c,
    "AC": lambda a, b, c: np.sign(a * b * c) * np.abs(a * b * c) ** 1/3,
    "AD": lambda a, b, c: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(b) / np.fft.fft2(c)).real,
    "AE": lambda a, b, c: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(c) / np.fft.fft2(b)).real,
    "AF": lambda a, b, c: np.fft.ifft2(np.fft.fft2(a) * np.fft.fft2(np.tanh(b)) / np.fft.fft2(np.tanh(c))).real,
    "AG": lambda a, b, c: np.fft.ifft2(np.fft.fft2(b) * np.fft.fft2(a) / np.fft.fft2(c)).real,
    "AH": lambda a, b, c: np.maximum(a, b, c),
    "AI": lambda a, b, c: np.minimum(a, b, c),
    "AJ": lambda a, b, c: ((a - b) ** 2 + (b - c) ** 2 + (c - a) ** 2) ** 1/2,
    "AK": lambda a, b, c: (a ** 2 + b ** 2 + c ** 2) ** 1/2,
    "AL": lambda a, b, c: a + (b - c) * 0.5,
    "AN": lambda a, b, c: np.take(np.fft.ifft(np.fft.fft(np.take(b.flatten(), np.argsort(a.flatten()))) * np.fft.fft(np.take(c.flatten(), np.argsort(a.flatten()))) / a.shape[0]**2).real, np.argsort(np.argsort(a.flatten()))).reshape(a.shape),
    "AO": lambda a, b, c: np.take(np.fft.ifft(np.fft.fft(np.take(a.flatten(), np.argsort(b.flatten()))) * np.fft.fft(np.take(c.flatten(), np.argsort(b.flatten()))) / a.shape[0]**2).real, np.argsort(np.argsort(b.flatten()))).reshape(a.shape),
    "AP": lambda a, b, c: np.take(np.fft.ifft(np.fft.fft(np.take(a.flatten(), np.argsort(c.flatten()))) * np.fft.fft(np.take(b.flatten(), np.argsort(c.flatten()))) / a.shape[0]**2).real, np.argsort(np.argsort(c.flatten()))).reshape(a.shape),
    "AQ": lambda a, b, c: (a + b + c) / 3,
    "AR": lambda a, b, c: a * (1 - np.tanh(b)) + c * (1 + np.tanh(b)),
    "AS": lambda a, b, c: b * (1 - np.tanh(a)) + c * (1 + np.tanh(a)),
    "AT": lambda a, b, c: b * (1 - np.tanh(np.mean(a))) + c * (1 + np.tanh(np.mean(a))),
    "AU": lambda a, b, c: a @ b.T @ c / a.shape[0]**4,
    "AV": lambda a, b, c: a @ np.tanh(b.T @ c) / a.shape[0]**4,
    "AW": lambda a, b, c: a @ (np.tanh(b.T @ c) + 1) / a.shape[0]**4,
    "AX": lambda a, b, c: np.tanh(a @ b.T) @ c / a.shape[0]**4,
}


# --- end original function dicts ---

i0t = list(i0__.values())
i1t = list(i1_.values())
i2t = list(i2_.values())

len_i0 = len(i0t)
len_i1 = len(i1t)
len_i2 = len(i2t)

# -----------------------------
# 関数速度計測（i0, i1, i2 をすべて計測して確率分布 T を作る）
# -----------------------------
G = []
t2 = np.random.normal(0, 1, (96, 128))
for f in i0t:
    g = time.perf_counter()
    for j in range(100):
        f(t2) if f is not None else None
    G.append((time.perf_counter() - g) / 100)
    t2 = np.random.normal(0, 1, (96, 128))

for f in i1t:
    g = time.perf_counter()
    for j in range(100):
        f(t2, t2) if f is not None else None
    G.append((time.perf_counter() - g) / 100)
    t2 = np.random.normal(0, 1, (96, 128))

for f in i2t:
    g = time.perf_counter()
    for j in range(100):
        f(t2, t2, t2) if f is not None else None
    G.append((time.perf_counter() - g) / 100)
    t2 = np.random.normal(0, 1, (96, 128))

# normalize into probability per-function
T = np.asarray(1.0 / (np.array(G) ** 0.3 + 1e-12))
T = T / np.sum(T)
print(T)
print("function distribution T length:", T.shape)

# -----------------------------
# 定数（必要なら調整）
# -----------------------------
QUANT_BITS = 12   # a を何ビットで量子化するか（デフォルト 12bit -> 0..4095）
QUANT_SCALE = (1 << QUANT_BITS) - 1

# -----------------------------
# numba版 compute_used_nodes_numba （三項対応）
# -----------------------------
@njit(cache=True)
def compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1):
    N = G1.shape[0]
    used = np.zeros((N, MODELLEN), dtype=np.int8)
    stack = np.empty(MODELLEN, dtype=np.int32)
    for ind in range(N):
        top = 0
        start = MODELLEN - last_k
        if start < 4:
            start = 4
        for s in range(start, MODELLEN):
            stack[top] = s
            top += 1
        while top > 0:
            top -= 1
            n = stack[top]
            if used[ind, n] == 1:
                continue
            used[ind, n] = 1
            if n <= 2:
                continue
            func_id = int(G2[ind, n])
            # unary
            if func_id < len_i0:
                a = int(abs(G1[ind, n, 0]))
                if used[ind, a] == 0:
                    stack[top] = a
                    top += 1
            # binary
            elif func_id < (len_i0 + len_i1):
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                if used[ind, a] == 0:
                    stack[top] = a
                    top += 1
                if used[ind, b] == 0:
                    stack[top] = b
                    top += 1
            # ternary
            else:
                a = int(abs(G1[ind, n, 0]))
                b = int(abs(G1[ind, n, 1]))
                c = int(abs(G1[ind, n, 2]))
                if used[ind, a] == 0:
                    stack[top] = a
                    top += 1
                if used[ind, b] == 0:
                    stack[top] = b
                    top += 1
                if used[ind, c] == 0:
                    stack[top] = c
                    top += 1
        used[ind, 0] = 1
        used[ind, 1] = 1
        used[ind, 2] = 1
    return used

@njit(cache=True)
def _hash_insert_get_sid(key_table_keys, key_table_vals, key, next_sid, table_mask):
    h = (key ^ (key >> 33)) & table_mask
    while True:
        k = key_table_keys[h]
        if k == -1:
            key_table_keys[h] = key
            key_table_vals[h] = next_sid
            return next_sid, 1, next_sid + 1
        elif k == key:
            return key_table_vals[h], 0, next_sid
        else:
            h = (h + 1) & table_mask

# -----------------------------
# precompute_structs_numba に三項を追加、struct_alpha を返す
# -----------------------------
@njit(cache=True)
def precompute_structs_numba(G1, G2, G3, len_i0, len_i1, len_i2, last_k=10):
    N = G1.shape[0]
    MODELLEN = G1.shape[1]
    used = compute_used_nodes_numba(G1, G2, MODELLEN, last_k, len_i0, len_i1)

    total_used = 0
    for ind in range(N):
        for node in range(MODELLEN):
            if used[ind, node] == 1:
                total_used += 1

    size = 1
    while size < total_used * 4:
        size <<= 1
    table_mask = size - 1
    key_table_keys = np.empty(size, dtype=np.int64)
    key_table_vals = np.full(size, -1, dtype=np.int32)
    for i in range(size):
        key_table_keys[i] = -1

    max_S = total_used + 8
    struct_type = np.empty(max_S, dtype=np.int32)
    struct_func = np.empty(max_S, dtype=np.int32)
    struct_ch1 = np.empty(max_S, dtype=np.int32)
    struct_ch2 = np.empty(max_S, dtype=np.int32)
    struct_ch3 = np.empty(max_S, dtype=np.int32)
    struct_alpha = np.empty(max_S, dtype=np.float32)

    # reserve sid 0,1,2 for inputs (R,G,B)
    next_sid = 4
    for sid in range(4):
        struct_type[sid] = 0
        struct_func[sid] = -1
        struct_ch1[sid] = -1
        struct_ch2[sid] = -1
        struct_ch3[sid] = -1
        struct_alpha[sid] = 0.0

    node_structs = np.full((N, MODELLEN), -1, dtype=np.int32)

    pairs_ind = np.empty(total_used + 1, dtype=np.int32)
    pairs_node = np.empty(total_used + 1, dtype=np.int32)
    pair_pos = 0

    for node in range(MODELLEN):
        for ind in range(N):
            if used[ind, node] == 0:
                continue
            if node <= 2:
                node_structs[ind, node] = node
                pairs_ind[pair_pos] = ind
                pairs_node[pair_pos] = node
                pair_pos += 1
                continue
            func_id = int(G2[ind, node])
            a_quant = int(np.floor(G3[ind, node] * QUANT_SCALE + 0.5))
            if a_quant < 0:
                a_quant = 0
            if a_quant > QUANT_SCALE:
                a_quant = QUANT_SCALE

            # unary
            if func_id < len_i0:
                a = int(abs(G1[ind, node, 0]))
                child_sid = node_structs[ind, a]
                key = ((1 << 60) | (func_id << 36) | ((child_sid & 0xFFFFF) << 16) | (a_quant & 0xFFFF))
            # binary
            elif func_id < (len_i0 + len_i1):
                bi = func_id - len_i0
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1]))
                child_a_sid = node_structs[ind, a]
                child_b_sid = node_structs[ind, b]
                key = ((2 << 60) | (bi << 36) | ((child_a_sid & 0xFFFFF) << 16) | (child_b_sid & 0xFFF))
                key = key ^ (a_quant & 0xFFFF)
            # ternary
            else:
                ci = func_id - (len_i0 + len_i1)
                a = int(abs(G1[ind, node, 0])); b = int(abs(G1[ind, node, 1])); c = int(abs(G1[ind, node, 2]))
                child_a_sid = node_structs[ind, a]
                child_b_sid = node_structs[ind, b]
                child_c_sid = node_structs[ind, c]
                # pack into key (may overlap; acceptable)
                key = ((3 << 60) | (ci << 36) | ((child_a_sid & 0xFFFFF) << 16) | ((child_b_sid & 0xFFF) << 6) | (child_c_sid & 0x3F))
                key = key ^ (a_quant & 0xFFFF)

            sid, is_new, next_sid = _hash_insert_get_sid(key_table_keys, key_table_vals, key, next_sid, table_mask)
            if is_new == 1:
                if func_id < len_i0:
                    struct_type[sid] = 1
                    struct_func[sid] = func_id
                    struct_ch1[sid] = child_sid
                    struct_ch2[sid] = -1
                    struct_ch3[sid] = -1
                elif func_id < (len_i0 + len_i1):
                    struct_type[sid] = 2
                    struct_func[sid] = func_id - len_i0
                    struct_ch1[sid] = child_a_sid
                    struct_ch2[sid] = child_b_sid
                    struct_ch3[sid] = -1
                else:
                    struct_type[sid] = 3
                    struct_func[sid] = func_id - (len_i0 + len_i1)
                    struct_ch1[sid] = child_a_sid
                    struct_ch2[sid] = child_b_sid
                    struct_ch3[sid] = child_c_sid
                struct_alpha[sid] = float(a_quant) / float(QUANT_SCALE)
            node_structs[ind, node] = sid
            pairs_ind[pair_pos] = ind
            pairs_node[pair_pos] = node
            pair_pos += 1

    S = next_sid
    struct_type = struct_type[:S].copy()
    struct_func = struct_func[:S].copy()
    struct_ch1 = struct_ch1[:S].copy()
    struct_ch2 = struct_ch2[:S].copy()
    struct_ch3 = struct_ch3[:S].copy()
    struct_alpha = struct_alpha[:S].copy()

    counts = np.zeros(S, dtype=np.int32)
    for p in range(pair_pos):
        sid = node_structs[pairs_ind[p], pairs_node[p]]
        counts[sid] += 1
    idxs = np.empty(S + 1, dtype=np.int32)
    idxs[0] = 0
    for s in range(S):
        idxs[s+1] = idxs[s] + counts[s]
    M = idxs[-1]
    struct_to_nodes_pair = np.empty((M, 2), dtype=np.int32)
    write_pos = idxs[:-1].copy()
    for p in range(pair_pos):
        ind = pairs_ind[p]
        node = pairs_node[p]
        sid = node_structs[ind, node]
        pos = write_pos[sid]
        struct_to_nodes_pair[pos, 0] = ind
        struct_to_nodes_pair[pos, 1] = node
        write_pos[sid] += 1

    return node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, struct_to_nodes_pair

# -----------------------------
# topo_sort_structs_numba_from_arrays（三項に対応）
# -----------------------------
@njit(cache=True)
def topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3):
    S = struct_type.shape[0]
    parent_count = np.zeros(S, dtype=np.int32)
    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            c = struct_ch1[sid]
            if c >= 0:
                parent_count[c] += 1
        elif t == 2:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]
            if c1 >= 0:
                parent_count[c1] += 1
            if c2 >= 0:
                parent_count[c2] += 1
        elif t == 3:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]; c3 = struct_ch3[sid]
            if c1 >= 0:
                parent_count[c1] += 1
            if c2 >= 0:
                parent_count[c2] += 1
            if c3 >= 0:
                parent_count[c3] += 1

    tot = 0
    offsets = np.empty(S, dtype=np.int32)
    for i in range(S):
        offsets[i] = tot
        tot += parent_count[i]
    parent_buf = np.empty(tot, dtype=np.int32)

    for i in range(S):
        parent_count[i] = 0

    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            c = struct_ch1[sid]
            if c >= 0:
                idx = offsets[c] + parent_count[c]
                parent_buf[idx] = sid
                parent_count[c] += 1
        elif t == 2:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]
            if c1 >= 0:
                idx = offsets[c1] + parent_count[c1]
                parent_buf[idx] = sid
                parent_count[c1] += 1
            if c2 >= 0:
                idx = offsets[c2] + parent_count[c2]
                parent_buf[idx] = sid
                parent_count[c2] += 1
        elif t == 3:
            c1 = struct_ch1[sid]; c2 = struct_ch2[sid]; c3 = struct_ch3[sid]
            if c1 >= 0:
                idx = offsets[c1] + parent_count[c1]
                parent_buf[idx] = sid
                parent_count[c1] += 1
            if c2 >= 0:
                idx = offsets[c2] + parent_count[c2]
                parent_buf[idx] = sid
                parent_count[c2] += 1
            if c3 >= 0:
                idx = offsets[c3] + parent_count[c3]
                parent_buf[idx] = sid
                parent_count[c3] += 1

    indeg = np.zeros(S, dtype=np.int32)
    for sid in range(S):
        t = struct_type[sid]
        if t == 1:
            if struct_ch1[sid] >= 0:
                indeg[sid] += 1
        elif t == 2:
            if struct_ch1[sid] >= 0:
                indeg[sid] += 1
            if struct_ch2[sid] >= 0:
                indeg[sid] += 1
        elif t == 3:
            if struct_ch1[sid] >= 0:
                indeg[sid] += 1
            if struct_ch2[sid] >= 0:
                indeg[sid] += 1
            if struct_ch3[sid] >= 0:
                indeg[sid] += 1

    q = np.empty(S, dtype=np.int32)
    ql = 0; qr = 0
    for s in range(S):
        if indeg[s] == 0:
            q[qr] = s; qr += 1
    out = np.empty(S, dtype=np.int32)
    out_len = 0
    while ql < qr:
        s = q[ql]; ql += 1
        out[out_len] = s; out_len += 1
        start = offsets[s]
        end = offsets[s] + parent_count[s]
        for pidx in range(start, end):
            p = parent_buf[pidx]
            indeg[p] -= 1
            if indeg[p] == 0:
                q[qr] = p; qr += 1
    if out_len < S:
        res = np.empty(out_len, dtype=np.int32)
        for i in range(out_len):
            res[i] = out[i]
        return res
    return out

# -----------------------------
# batch_exec_structured_py（三項に対応）
# -----------------------------
def batch_exec_structured_py(input_arr,
                             node_structs,
                             struct_type,
                             struct_func,
                             struct_ch1,
                             struct_ch2,
                             struct_ch3,
                             struct_alpha,
                             topo,
                             last_k=3,
                             restrict=True):
    N = node_structs.shape[0]
    MODELLEN = node_structs.shape[1]
    if input_arr.ndim == 3:
        H, W, C = input_arr.shape
    elif input_arr.ndim == 2:
        H, W = input_arr.shape
        C = 1
    else:
        raise RuntimeError("unsupported input_arr shape: %r" % (input_arr.shape,))
    S = struct_type.shape[0]

    if restrict:
        needed = np.zeros(S, dtype=np.bool_)
        q = []
        start_nodes = range(max(0, MODELLEN - last_k), MODELLEN)
        for ind in range(N):
            for ln in start_nodes:
                sid = int(node_structs[ind, ln])
                if sid >= 0 and not needed[sid]:
                    needed[sid] = True; q.append(sid)
        qi = 0
        while qi < len(q):
            s = q[qi]; qi += 1
            if s < len(struct_ch1):
                c1 = int(struct_ch1[s]); c2 = int(struct_ch2[s]); c3 = int(struct_ch3[s])
            else:
                c1 = -1; c2 = -1; c3 = -1
            if c1 >= 0 and not needed[c1]:
                needed[c1] = True; q.append(c1)
            if c2 >= 0 and not needed[c2]:
                needed[c2] = True; q.append(c2)
            if c3 >= 0 and not needed[c3]:
                needed[c3] = True; q.append(c3)
        if S >= 4:
            needed[0] = True; needed[1] = True; needed[2] = True
    else:
        needed = np.ones(S, dtype=np.bool_)

    outputs = [None] * S
    if input_arr.ndim == 3:
        for j in range(input_arr.shape[-1]):
            outputs[j] = input_arr[:, :, j]
    else:
        outputs[0] = input_arr; outputs[1] = np.zeros_like(input_arr); outputs[2] = np.zeros_like(input_arr)

    i0_funcs = i0t; i1_funcs = i1t; i2_funcs = i2t

    for sid in topo:
        if sid < 0 or sid >= S: continue
        if not needed[sid]: continue
        if sid <= 2: continue
        t = int(struct_type[sid])
        alpha = float(struct_alpha[sid]) if sid < len(struct_alpha) else 0.0
        if t == 1:
            func_id = int(struct_func[sid])
            child_sid = int(struct_ch1[sid])
            a = outputs[child_sid]
            base = i0_funcs[func_id](a)
            out = (1.0 - alpha) * base + alpha * a
            outputs[sid] = out
        elif t == 2:
            func_id = int(struct_func[sid])
            c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid])
            a = outputs[c1]; b = outputs[c2]
            base = i1_funcs[func_id](a, b)
            out = (1.0 - alpha) * base + alpha * a
            outputs[sid] = out
        elif t == 3:
            func_id = int(struct_func[sid])
            c1 = int(struct_ch1[sid]); c2 = int(struct_ch2[sid]); c3 = int(struct_ch3[sid])
            a = outputs[c1]; b = outputs[c2]; c = outputs[c3]
            base = i2_funcs[func_id](a, b, c)
            out = (1.0 - alpha) * base + alpha * a
            outputs[sid] = out
        else:
            raise RuntimeError("unknown struct type: %r" % t)

    last_nodes = list(range(max(0, MODELLEN - last_k), MODELLEN))
    logits_list = []
    zero_like = np.zeros((H, W))
    for ind in range(N):
        stack = []
        for ln in last_nodes:
            sid = int(node_structs[ind, ln])
            if sid >= 0 and outputs[sid] is not None:
                stack.append(outputs[sid])
            else:
                stack.append(zero_like)
        stacked = np.stack(stack, axis=0)
        logits_list.append(stacked)
    return logits_list

# -----------------------------
# メイン側で使う遺伝子の形を三項に合わせる
# -----------------------------
MODELLEN = 500000
GENES1 = []
GENES2 = []
GENES3 = []


for p in range(144):
    GENES1.append(np.abs((1-np.random.uniform(0, 1, (MODELLEN, 3)) ** 1.5) * (np.arange(MODELLEN)[:, None])))
    GENES2.append(np.random.choice(len(i0t) + len(i1t) + len(i2t), (MODELLEN), p=T))
    GENES3.append(np.random.uniform(0, 1, (MODELLEN)))
    # G1 は (MODELLEN, 3) にして三つの子ノード参照を持たせる

if(os.path.exists("dats.npz")):
    data = np.load("dats.npz")
    GENES1 = [_ for _ in data["genes1"]]
    GENES2 = [_ for _ in data["genes2"]]
    GENES3 = [_ for _ in data["genes3"]]

#datas = [trainset[np.random.randint(0, len(trainset)-1)] for j in range(64)]
#test_datas = [testset[np.random.randint(0, len(testset)-1)] for j in range(2048)]
lp = 0

bestacc = 10
elites1 = []
elites2 = []
elites3 = []

accs = []
accs2 = []

G1 = np.stack([g.astype(np.int64) for g in GENES1], axis=0)   # shape (N, MODELLEN, 2)
G2 = np.stack([g.astype(np.int64) for g in GENES2], axis=0)   # shape (N, MODELLEN)
G3 = np.stack([g.astype(np.float32) for g in GENES3], axis=0)   # shape (N, MODELLEN)

import torch
from diffusers import DiffusionPipeline, AutoencoderKL

ae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16).to("cuda")

node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3, struct_alpha, idxs, struct_to_nodes_pair = \
    precompute_structs_numba(G1, G2, G3, len(i0t), len(i1t), len(i2t), last_k=4)

topo = topo_sort_structs_numba_from_arrays(struct_type, struct_ch1, struct_ch2, struct_ch3)

print("topology calculated")

for i in range(1):
    img = batch_exec_structured_py(np.random.uniform(0, 1, (128, 96, 4)),
                                            node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3,
                                            struct_alpha, topo, last_k=4, restrict=True)

    for j in tqdm.tqdm(range(len(img))):
        cv2.imwrite(f"imgs/0_{j}_{i}.jpg", np.maximum(0, np.minimum(255, ae.decode(torch.tensor(img[j][None], dtype=torch.float16).to("cuda"))[0][0].cpu().detach().numpy().transpose((1, 2, 0)) * 255)))


NOW_ITER = 0
ELO_RATINGS = np.zeros(12*12)
VOTED_LIST = np.zeros(12*12)

regenerating = False

def regenerate():
    global GENES1, GENES2, GENES3, NOW_ITER, ELO_RATINGS, VOTED_LIST, regenerating
    if regenerating:
        return
    regenerating = True
    rank = np.argsort(-np.array(ELO_RATINGS))
    NGENES1 = []
    NGENES2 = []
    NGENES3 = []
    for i in range(12):
        NGENES1.append(deepcopy(GENES1[rank[i]]))
        NGENES2.append(deepcopy(GENES2[rank[i]]))
        NGENES3.append(deepcopy(GENES3[rank[i]]))
        for j in range(11):
            NGENE1 = deepcopy(GENES1[rank[i]])
            NGENE2 = deepcopy(GENES2[rank[i]])
            NGENE3 = deepcopy(GENES3[rank[i]])
            pos1 = np.random.randint(0, MODELLEN-2)
            pos2 = np.random.randint(pos1, MODELLEN-1)
            NGENE1[pos1:pos2] = np.copy(GENES1[rank[j]][pos1:pos2])
            NGENE2[pos1:pos2] = np.copy(GENES2[rank[j]][pos1:pos2])
            p = np.random.uniform(0, 1)
            NGENE3[pos1:pos2] = NGENE3[pos1:pos2] * p + GENES3[rank[j]][pos1:pos2] * (1-p)
            if(np.random.uniform(0, 1) < 0.05):
                for __ in range(np.random.randint(1, 2**np.random.randint(1, 14))):
                    pos = np.random.randint(4, MODELLEN-3)
                    NGENE3[pos] = np.random.uniform(0, 1)
            if(np.random.uniform(0, 1) < 0.01):
                pos1 = np.random.randint(0, MODELLEN-2)
                pos2 = np.random.randint(pos1, MODELLEN-1)
                NGENE3[pos1:pos2] = NGENE3[pos1:pos2] * np.random.uniform(0.2, 0.8) + np.random.normal(0, np.random.uniform(0, 1)) * np.random.uniform(0, 1)
            if(np.random.uniform(0, 1) < 0.005):
                NGENE3 = np.fft.ifft(np.fft.fft(NGENE3+0j, axis=0) * np.fft.fft(GENES3[rank[j]]+0j, axis=0) / np.fft.fft(GENES3[np.random.randint(0, len(GENES3))], axis=0), axis=0).real
            if(np.random.uniform(0, 1) < 0.005):
                NGENE3 = np.floor(NGENE3 + (GENES3[rank[j]] - GENES3[np.random.randint(0, len(GENES3))]) * np.random.uniform(0, 1.5))
            NGENE3 = np.maximum(np.minimum(NGENE3, 1), 0)
            if(np.random.uniform(0, 1) < 0.05):
                for __ in range(np.random.randint(1, 2**np.random.randint(1, 14))):
                    pos = np.random.randint(4, MODELLEN-3)
                    NGENE1[pos][np.random.randint(0, 3)] = np.random.randint(0, pos-1)
            if(np.random.uniform(0, 1) < 0.05):
                for __ in range(np.random.randint(1, 2**np.random.randint(2, 14))):
                    pos = np.random.randint(4, MODELLEN-3)
                    NGENE2[pos] = np.random.choice(len(i0t) + len(i1t) + len(i2t), p=T)
            if(np.random.uniform(0, 1) < 0.00015):
                NGENE1 = np.abs(np.random.uniform(0, 1, (MODELLEN, 3)) * (np.arange(MODELLEN)[:, None]))
            if(np.random.uniform(0, 1) < 0.00015):
                NGENE2 = np.random.choice(len(i0t) + len(i1t) + len(i2t), (MODELLEN), p=T)
            if(np.random.uniform(0, 1) < 0.003125):
                TT3 = 2**np.random.randint(1, 14)
                pos1 = np.random.randint(0, MODELLEN-2)
                pos2 = np.random.randint(pos1, MODELLEN)
                lv = np.random.randint(-TT3, TT3)
                NGENE1[pos1:pos2] = NGENE1[pos1:pos2] + lv
            if(np.random.uniform(0, 1) < 0.0003125):
                NGENE1 = np.floor(np.fft.ifft(np.fft.fft(NGENE1+0j, axis=0) * np.fft.fft(GENES1[rank[j]]+0j, axis=0) / np.fft.fft(GENES1[np.random.randint(0, len(GENES1))], axis=0), axis=0).real)
            if(np.random.uniform(0, 1) < 0.00015):
                NGENE1 = np.floor(NGENE1 + (GENES1[rank[j]] - GENES1[np.random.randint(0, len(GENES1))]) * np.random.uniform(0, 1.5))
            if(np.random.uniform(0, 1) < 0.00015):
                NGENE1 = np.floor(NGENE1 + (GENES1[rank[j]] - GENES1[np.random.randint(0, len(GENES1))]))
            if(np.random.uniform(0, 1) < 0.003125):
                TT3 = 2**np.random.randint(1, 14)
                pos1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                pos2 = np.random.randint(pos1, MODELLEN-TT3-1)
                pos3 = np.random.randint(-TT3, TT3)
                NGENE1[pos1-pos3:pos2-pos3] = NGENE1[pos1:pos2]
            if(np.random.uniform(0, 1) < 0.003125):
                TT3 = 2**np.random.randint(1, 14)
                pos1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                pos2 = np.random.randint(pos1, MODELLEN-TT3-1)
                pos3 = np.random.randint(-TT3, TT3)
                NGENE2[pos1-pos3:pos2-pos3] = NGENE2[pos1:pos2]
            if(np.random.uniform(0, 1) < 0.00625):
                TT3 = 2**np.random.randint(1, 14)
                pos1 = np.random.randint(TT3+1, MODELLEN-TT3-1)
                pos2 = np.random.randint(pos1, MODELLEN-TT3-1)
                pos3 = np.random.randint(-TT3, TT3)
                NGENE1[pos1-pos3:pos2-pos3] = NGENE1[pos1:pos2]
                NGENE2[pos1-pos3:pos2-pos3] = NGENE2[pos1:pos2]
            NGENE1 = np.maximum(NGENE1, 0)
            NGENE1 = np.minimum(NGENE1, np.maximum((np.arange(MODELLEN)-2)[:, None], 0))
            NGENES1.append(np.abs(np.copy(NGENE1)))
            NGENES2.append(np.maximum(np.minimum(NGENE2, len(i0t) + len(i1t) + len(i2t)), 0))
            NGENES3.append(np.copy(NGENE3))
            del NGENE1, NGENE2, NGENE3
    GENES1 = NGENES1
    GENES2 = NGENES2
    GENES3 = NGENES3
    for i in range(1):
        img = batch_exec_structured_py(np.random.uniform(0, 1, (128, 96, 4)),
                                                node_structs, struct_type, struct_func, struct_ch1, struct_ch2, struct_ch3,
                                                struct_alpha, topo, last_k=4, restrict=True)

        for j in tqdm.tqdm(range(len(img))):
            cv2.imwrite(f"imgs/{NOW_ITER+1}_{j}_{i}.jpg", np.maximum(0, np.minimum(255, ae.decode(torch.tensor(img[j][None], dtype=torch.float16).to("cuda"))[0][0].cpu().detach().numpy().transpose((1, 2, 0)) * 255)))
    NOW_ITER += 1
    gc.collect()
    ELO_RATINGS = np.zeros(144)
    VOTED_LIST = np.zeros(144)
    regenerating = False
    np.savez('dats.npz', genes1=GENES1, genes2=GENES2, genes3=GENES3)
    return

app = Flask(__name__, static_folder="imgs")

@app.route("/pair", methods=["GET"])
def get_pair():
    global NOW_ITER, VOTED_LIST
    i = np.random.randint(0, 1)
    j = np.random.randint(0, 12*12)
    j2 = np.random.randint(0, 12*12)
    while j == j2:
        j2 = np.random.randint(0, 12*12)
    return jsonify({
        "left": {"id": j, "url": f"/imgs/{NOW_ITER}_{j}_{i}.jpg", "score": ELO_RATINGS[j]},
        "right": {"id": j2, "url": f"/imgs/{NOW_ITER}_{j2}_{i}.jpg", "score": ELO_RATINGS[j2]},
        "iteration": NOW_ITER,
        "iteration_2": np.sum(VOTED_LIST)//2,
    })

@app.route("/", methods=["GET"])
def index():
    return open('index.html', 'r').read()

@app.route("/vote", methods=["POST"])
def vote():
    global ELO_RATINGS, VOTED_LIST
    data = request.get_json(force=True)
    if not data or "winner_id" not in data or "loser_id" not in data or int(data["winner_id"]) > 12*12 or int(data["loser_id"]) > 12*12:
        return jsonify({"error":"invalid payload"}), 400
    a = ELO_RATINGS[int(data["winner_id"])]
    b = ELO_RATINGS[int(data["loser_id"])]
    Wab = 32 / (10 ** ((a - b) / 400) + 1)

    ELO_RATINGS[int(data["winner_id"])] += Wab
    ELO_RATINGS[int(data["loser_id"])] -= Wab
    VOTED_LIST[int(data["winner_id"])] += 1
    VOTED_LIST[int(data["loser_id"])] += 1

    print(ELO_RATINGS)
    print(VOTED_LIST)

    if(np.median(VOTED_LIST) >= 5 and np.min(VOTED_LIST) >= 1 and np.sum(VOTED_LIST) >= 800):
        regenerate()
    return jsonify({"status":"ok"})

app.run(host="0.0.0.0", port=8080, debug=False)