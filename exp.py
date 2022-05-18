#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Solves Torus Puzzle on Exponential Idle on Android (no root)"""
__author__ = "Sunny Patel <github.com/laughdonor>"
from __future__ import annotations
from dataclasses import dataclass, astuple
from itertools import groupby
from PIL import Image
from pprint import pprint
from pytesseract import pytesseract, image_to_string, image_to_data, Output
from sys import exit
from time import sleep
from typing import List
import numpy as np
import cv2, os, torus  # Torus solver from https://codegolf.stackexchange.com/a/172852/58557

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
grid = Rect(x0=57, y0=896, x1=1041, y1=1880)
debug = False
tess_config = '--dpi 420 --psm 6'   # Use `adb shell wm density` to get the dpi
pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# In[3]:


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def capture(temp_file: str ="scr.png") -> Image:
    os.system(r"adb exec-out screencap -p | perl -pe 's/\x0D\x0A/\x0A/g' > " + temp_file)
    grey = cv2.bitwise_not(cv2.cvtColor(cv2.imread(temp_file), cv2.COLOR_BGR2GRAY))
    if not debug:
        os.remove(temp_file)
    return Image.fromarray(cv2.threshold(grey, 152, 255, cv2.THRESH_BINARY)[1])

def solve(image: Image) -> (List[(int, str, int)], int):
    t = image_to_string(image.crop(astuple(grid)), config=f"{tess_config} --tessdata-dir ./tess -l digits").strip()
    rows = t.count("\n") + 1
    M = np.fromstring(t, sep=" ", dtype=int).reshape(-1, rows)

    if len(np.unique(M)) != rows ** 2:
        print(f"\n\n\nERROR: Didn't recognize every number: {len(np.unique(M))}\n\n\n")
        pprint(M)
        exit()
    return [(int(r), d, len(list(g))) for (d, r), g in groupby(torus.f(M.tolist()))], rows

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

def tap(data: dict, idx: int) -> List[str]:
    return [f"input touchscreen tap {data['left'][idx]} {data['top'][idx]}"]

def send_commands(commands: List[str]):
    for batchset in batch(commands, 16):  # 16 is used because of 1024 character limit of adb commands
        command = " && sleep 0.1 && ".join(batchset)
        if debug:
            print(f"adb shell \"{command}\"")
        os.system(f"adb shell \"{command}\"")
        sleep(0.1)
    sleep(1)


# In[4]:


while True:
    image = capture()
    text = image_to_string(image, config=tess_config).strip()
    data = image_to_data(image, config=tess_config, output_type=Output.DICT)

    if "Give Up" in text:
        send_commands(swipes(*solve(image)))
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

