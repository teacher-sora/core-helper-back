from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict, Counter

import numpy as np
import cv2
import os
import gc
import math
import asyncio

import time

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
  allow_credentials=True
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def health():
  print("Health Checked")
  return { "status": "ok" }

@app.post("/core-helper/")
async def core_helper(images: list[UploadFile] = File(...), selected_job_class: str = Form(...)):
  try:
    processed = await asyncio.wait_for(process(images, selected_job_class), timeout=12.5)
    if processed.get("message") is not None:
      return JSONResponse(content={
          "success": processed.get("success", False),
          "message": processed.get("message", "")
      })
    elif processed.get("core_skill_names") is not None:
      return JSONResponse(content={
          "success": processed.get("success", False),
          "core_skill_names": processed.get("core_skill_names", [])
      })
  except asyncio.TimeoutError:
    return JSONResponse(content={
      "success": False,
      "message": "요청 시간이 초과되었습니다."
    })
  except Exception as e:
    import traceback
    traceback.print_exc()

    return JSONResponse(status_code=500, content={
      "success": False
    })

async def process(images: list[UploadFile] = File(...), selected_job_class: str = Form(...)):
  start_time = time.time()
  print(f"요청 - 직업: [{selected_job_class}], 이미지: [{len(images)}]")

  # 경로 설정
  base_path = "static"
  job_class_path = os.path.join(base_path, "skills", selected_job_class)
  skills = get_job_skills(job_class_path)

  displays = []
  for image in images:
    content = await image.read()
    np_img = np.frombuffer(content, np.uint8)

    display = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    displays.append(display)

    del content
    del np_img
    del display
    gc.collect()

  icons = []
  for display in displays:
    cores = find_core_candidates(display)
    # print(f"cores: {len(cores)}")
    if not cores:
      continue

    enhanced_cores = extract_enhanced_core_candidates(cores)
    # print(f"enhanced_cores: {len(enhanced_cores)}")
    if not enhanced_cores:
      continue

    core_icons = extract_core_icon_candidates(enhanced_cores)
    # print(f"core_icons: {len(core_icons)}")
    if not core_icons:
      continue

    valid_core_icons = filter_valid_core_icons(core_icons)
    # print(f"valid_core_icons: {len(valid_core_icons)}")
    if valid_core_icons:
      icons.extend(valid_core_icons)

    del cores
    del enhanced_cores
    del core_icons
    del valid_core_icons
    gc.collect()
  find_cores = time.time()
  print(f"경과 시간[코어 탐색]: {find_cores - start_time:.3f}초, 탐색된 코어: {len(icons)}개")

  detected_cores = []
  for icon in icons:
    detected_skill_names = analyze_icon(icon, skills)
    if detected_skill_names:
      detected_cores.append(detected_skill_names)
  analyze_cores = time.time()
  print(f"경과 시간[코어 분석]: {analyze_cores - find_cores:.3f}초, 분석된 코어: {len(detected_cores)}개")
  print(f"소요 시간: {analyze_cores - start_time:.3f}초")

  if not detected_cores:
    return {
      "success": False,
      "message": "이미지에서 쓸만한 코어가 발견되지 않았어요.\n다시 한번 확인해 주세요."
    }
  else:
    return {
      "success": True,
      "core_skill_names": detected_cores
    }

def get_job_skills(job_class_path):
  skills = []
  for file_name in os.listdir(job_class_path):
    if file_name.lower().endswith(".png"):
      path = os.path.join(job_class_path, file_name)
      icon = cv2.imread(path)
      outlined_icon = outline_icon(icon)
      name = file_name[:-4]
      skills.append({"icon": outlined_icon, "name": name})
  return skills

def find_core_candidates(display, color = [180, 255, 50]):
  lower_color = np.array([0, 0, 0])
  upper_color = np.array(color)

  hsv = cv2.cvtColor(display, cv2.COLOR_BGR2HSV)
  smoothed = cv2.bilateralFilter(hsv, d=9, sigmaColor=20, sigmaSpace=20)
  masked = cv2.inRange(smoothed, lower_color, upper_color)
  contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  core_candidates = []
  padding = 20
  height, width = display.shape[:2]
  for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # 크기가 100보다 작으면 스킵
    area = w * h
    if area < 100:
      continue

    # 사각형이 아니면 스킵
    aspect_ratio = float(w) / h
    if not (0.8 <= aspect_ratio <= 1.2):
      continue

    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    x2 = min(x + w + padding, width)
    y2 = min(y + h + padding, height)

    crop = display[y1:y2, x1:x2]
    core_candidates.append(crop)

  if (color == [180, 255, 50]) and (not core_candidates):
    return find_core_candidates(display, color=[180, 255, 30])
  else:
    return core_candidates

def extract_enhanced_core_candidates(cores):
  lower_color = np.array([0, 0, 0])
  upper_color = np.array([94, 115, 113])

  enhanced_core_candidates = []
  for core in cores:
    masked = cv2.inRange(core, lower_color, upper_color)

    h, w = masked.shape
    x = w // 6
    y = (h // 2) - 5

    # 강화코어 후보(흰색 많음)
    counter = Counter()
    for i in range(10):
      counter[f"{masked[y+i, x]}"] += 1
    if counter['255'] > counter['0']:
      enhanced_core_candidates.append(core)
  return enhanced_core_candidates

def extract_core_icon_candidates(cores, upper_color = np.array([180, 255, 60])):
  size = 32

  lower_color = np.array([0, 0, 0])

  core_icon_candidates = []
  for core in cores:
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    smoothed = cv2.bilateralFilter(hsv, d=9, sigmaColor=20, sigmaSpace=20)
    masked = cv2.inRange(smoothed, lower_color, upper_color)

    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
      continue

    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    crop = core[y:y+h, x:x+w]
    crop = cv2.resize(crop, (size, size))
    core_icon_candidates.append(crop)
  return core_icon_candidates

def filter_valid_core_icons(icons, upper_color = np.array([180, 255, 60])):
  lower_color = np.array([0, 0, 0])

  core_icons = []
  for icon in icons:
    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    smoothed = cv2.bilateralFilter(hsv, d=9, sigmaColor=20, sigmaSpace=20)
    masked = cv2.inRange(smoothed, lower_color, upper_color)
    edges = np.concatenate([masked[0, :], masked[-1, :], masked[:, 0], masked[:, -1]])

    # 스킬 후보(모서리에 흰색이 62 ~ 124)
    counter = Counter()
    counter.update(map(str, edges))
    if 62 <= counter["255"] <= 124:
      core_icons.append(icon)
  return core_icons

def outline_icon(icon):
  size = 32
  icon = cv2.resize(icon, (size, size))

  outer_color = [221, 221, 204]
  inner_color = [255, 255, 255]
  border_color = [0, 0, 0]

  def draw_horizontal(y, color):
    icon[y, 1:31] = color

  def draw_vertical(x, color):
    icon[2:30, x] = color

  # 바깥 테두리
  draw_horizontal(1, outer_color)   # 위
  draw_horizontal(30, outer_color)  # 아래

  # 안쪽 테두리
  draw_horizontal(2, inner_color)   # 위
  draw_horizontal(29, inner_color)  # 아래

  # 바깥 테두리
  draw_vertical(1, outer_color)     # 왼쪽
  draw_vertical(30, outer_color)    # 오른쪽

  # 안쪽 테두리
  draw_vertical(2, inner_color)     # 왼쪽
  draw_vertical(29, inner_color)    # 오른쪽

  # 외곽 경계선
  icon[0, 0:32] = border_color       # 최상단
  icon[31, 0:32] = border_color      # 최하단
  icon[0:32, 0] = border_color       # 최좌측
  icon[0:32, 31] = border_color      # 최우측

  return icon

def split_icon(icon):
  size = 32
  icon = cv2.resize(icon, (size, size))

  triangle_mask = np.zeros((size, size), dtype=np.uint8)
  pts = np.array([[0, 0], [31, 0], [16, 16]], np.int32)
  pts = pts.reshape((-1, 1, 2))
  cv2.fillPoly(triangle_mask, [pts], 255)

  canvas1, canvas2, canvas3 = (np.zeros_like(icon) for _ in range(3))
  for y in range(size):
    for x in range(size):
      if triangle_mask[y, x] == 0:
        (canvas1 if x < 16 else canvas3)[y, x] = icon[y, x]
  for center in range(3):
    canvas2[:, :, center] = np.where(triangle_mask == 255, icon[:, :, center], canvas2[:, :, center])
  return [canvas1, canvas2, canvas3]

def merge_icon_parts(parts):
  size = 32

  triangle_mask = np.zeros((size, size), dtype=np.uint8)
  pts = np.array([[0, 0], [31, 0], [16, 16]], np.int32)
  pts = pts.reshape((-1, 1, 2))
  cv2.fillPoly(triangle_mask, [pts], 255)

  canvas = np.zeros_like(parts[0])
  for y in range(size):
    for x in range(size):
      if triangle_mask[y, x] == 0:
        (canvas if x < 16 else canvas)[y, x] = (parts[0] if x < 16 else parts[2])[y, x]
  for center in range(3):
    canvas[:, :, center] = np.where(triangle_mask == 255, parts[1][:, :, center], canvas[:, :, center])
  return canvas

def mask_icon(sample, icon):
  mask = np.any(sample != 0, axis = -1)
  canvas = np.zeros_like(sample)
  canvas[mask] = icon[mask]
  return canvas

def analyze_icon(icon, skills):
  icon_parts = split_icon(icon)
  matched_skills = []
  for icon_part in icon_parts:
    match_results = []
    for skill in skills:
      if matched_skills:
        skill_names = [skill["name"] for skill in matched_skills]
        if skill["name"] in skill_names:
          continue
      masked = mask_icon(icon_part, skill["icon"])
      result = cv2.matchTemplate(icon_part, masked, cv2.TM_CCOEFF_NORMED)
      _, max_val, _, _ = cv2.minMaxLoc(result)
      match_results.append({"score": max_val, "skill": skill})
    skills = [result["skill"] for result in match_results]
    scores = [result["score"] for result in match_results]
    max_score = max(scores)
    max_score_index = scores.index(max_score)
    matched_skills.append(skills[max_score_index])
  
  skill_parts = [skill["icon"] for skill in matched_skills]
  skill_names = [skill["name"] for skill in matched_skills]
  merged_icon = merge_icon_parts(skill_parts)
  outlined_icon = outline_icon(merged_icon)

  result = cv2.matchTemplate(icon, outlined_icon, cv2.TM_CCOEFF_NORMED)
  _, max_val, _, _ = cv2.minMaxLoc(result)

  detected_skill_names = skill_names if max_val > 0.625 else []
  return detected_skill_names