#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Solves Torus Puzzle on Exponential Idle on Android (no root)"""
__author__ = "Sunny Patel <github.com/laughdonor>"
from __future__ import annotations
from dataclasses import dataclass, astuple
from itertools import groupby
import traceback
from PIL import Image
from pprint import pprint
from pytesseract import pytesseract, image_to_string, image_to_data, Output
from sys import exit
from time import sleep
from typing import List
import numpy as np
import cv2, os, torus, arrow  # Torus solver from https://codegolf.stackexchange.com/a/172852/58557

error_taps = np.array([[ 127., 1145.],
       [ 127., 1145.],
       [ 127., 1145.],
       [ 127., 1145.],
       [ 127., 1627.],
       [ 127., 1627.],
       [ 266., 1064.],
       [ 266., 1064.],
       [ 266., 1064.],
       [ 266., 1386.]])

# Requires Python 3.7+
@dataclass
class Rect:
    x0: int
    y0: int
    x1: int
    y1: int
        
    def cell(self, size) -> Rect:
        dx, dy = int((self.x1 - self.x0) / size), int((self.y1 - self.y0) / size)
        return Rect(x0=int(self.x0 + dx // 2), y0=int(self.y0 + dy // 2), x1=dx, y1=dy)
    def __getitem__(cls, x):
        return getattr(cls, x)
    def __setitem__(cls, x, value):
        return setattr(cls, x, value)


# In[2]:


# Phone specific coordinates of grid area. Change debug variable and get coordinates from temp_file (use paint)
grid = Rect(x0=57, y0=896, x1=1041, y1=1940)
hex_grid = {'offset':[[545,1386]], 'scale':1182}
debug = False
tess_config = '--dpi 440 --psm 6'   # Use `adb shell wm density` to get the dpi


# In[3]:


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def capture(temp_file: str ="scr.png") -> Image:
    os.system("adb shell screencap -p > " + temp_file)
    grey = cv2.bitwise_not(cv2.cvtColor(cv2.imread(temp_file), cv2.COLOR_BGR2GRAY))
    if not debug:
        os.remove(temp_file)
    return Image.fromarray(cv2.threshold(grey, 152, 255, cv2.THRESH_BINARY)[1])

def split_im(image: Image):
    im = np.array(image.crop(astuple(grid)))

    imgheight=im.shape[0]
    imgwidth=im.shape[1]

    y1 = 0
    M = imgheight
    N = imgwidth//7
    tiles = []
    for y in range(0,imgheight,M):
        for x in range(0, imgwidth, N):
            y1 = y + M
            x1 = x + N
            tiles.append(Image.fromarray(im[y:y+M,x:x+N]))

            # cv2.rectangle(im, (x, y), (x1, y1), (0, 255, 0))
            # cv2.imwrite(str(x) + '_' + str(y)+".png",tiles)
    return tiles
    

def solve(image: Image):
    cols = []
    for im in split_im(image):
        cols.append(image_to_string(im, config=f"{tess_config} -c tessedit_char_whitelist=0123456789").replace('\n\n','\n'))
    arr = [row.split('\n')[:-1] for row in cols[:7]]
    arr = [[int(i) for i in j] for j in arr]

    print('grid read')

    rectangle = arrow.make_grid(arr)
    taps = arrow.solve(rectangle)

    taps = taps/(3) * hex_grid['scale']/2

    taps = taps + hex_grid['offset']
    taps = np.around(taps)

    print('input sequence assembled')

    return taps

def swipes(steps: List[(int, str, int)], rows: int) -> List[str]:
    print(f"Solving with {len(steps)} swipes")
    output, cell = [], grid.cell(rows)

    for index, direction, count in steps:
        coord = Rect(x0=cell.x0, y0=cell.y0, x1=cell.x0, y1=cell.y0)
        if direction in "LR":
            coord[f'x{"LR".index(direction)}'] += count * cell.x1
            coord.y0 += index * cell.y1
            coord.y1 = coord.y0
        elif direction in "UD":
            coord[f'y{"UD".index(direction)}'] += count * cell.y1
            coord.x0 += index * cell.x1
            coord.x1 = coord.x0

        output.append(f"input touchscreen swipe {coord.x0} {coord.y0} {coord.x1} {coord.y1} 150")
    return output

def tap_puzzle(data) -> List[str]:
    return [f"input touchscreen tap {d[0]} {d[1]}" for d in data]

def tap(data: dict, idx: int) -> List[str]:
    return [f"input touchscreen tap {data['left'][idx]} {data['top'][idx]}"]

def send_commands(commands: List[str]):
    for batchset in batch(commands, 16):  # 16 is used because of 1024 character limit of adb commands
        command = " && ".join(batchset)
        if debug:
            print(f"adb shell \"{command}\"")
        os.system(f"adb shell \"{command}\"")
        sleep(0.1)


# In[4]:


while True:
    image = capture()
    text = image_to_string(image, config=tess_config).strip()
    data = image_to_data(image, config=tess_config, output_type=Output.DICT)

    if "Give Up" in text:
        try:
            taps = solve(image)
            send_commands(tap_puzzle(taps))
        except:
            traceback.print_exc()
            send_commands(tap_puzzle(error_taps))
            continue
    elif "Claim" in data['text']:
        pprint("Complete!")
        idx = data['text'].index("Claim")
        send_commands(tap(data, idx))
    elif "Play Torus Puzzle" in text:
        idx = data['text'].index("Torus")
        send_commands(tap(data, idx))
    elif "Select Difficulty" in text:
        idx = data['text'].index("Hard")
        send_commands(tap(data, idx))
    else:
        pprint("Required Text not Found!")
        pprint(text)
        pprint(data['text'])
        break


# %%
