from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO
from collections import defaultdict, Counter
from itertools import combinations, permutations

import numpy as np
import cv2
import os
import gc
import math

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

@app.post("/core-helper/")
async def core_helper(images: list[UploadFile] = File(...), selected_job_class: str = Form(...)):
  try:
    print(f"요청 - 직업: [{selected_job_class}], 이미지: [{len(images)}]")
    start_time = time.time()

    displays = []

    for image in images:
      # 이미지 읽어오기
      content = await image.read()
      npimg = np.frombuffer(content, np.uint8)
      
      # 이미지 추가
      display = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
      displays.append(display)

    # 경로 설정
    base_path = "static"
    selected_job_class_path = os.path.join(base_path, "skills", selected_job_class)
    decompose_tab_template = cv2.imread(os.path.join(base_path, "templates", "decompose-tab.png"))
    empty_core_template = cv2.imread(os.path.join(base_path, "templates", "empty-core.png"))

    generated_core_skills = generate_core_skills(selected_job_class_path)

    generate_time = time.time()
    print(f"경과 시간[코어 생성]: {generate_time - start_time:.3f}초")

    core_skill_names = []

    decompose_tabs = get_decompose_tabs(displays)

    model_time = time.time()
    print(f"경과 시간[모델 실행]: {model_time - generate_time:.3f}초")

    for decompose_tab in decompose_tabs:
      cores = get_cores(decompose_tab, empty_core_template)
      if len(cores) == 0:
        continue

      enhanced_cores = get_enhanced_cores(cores)
      if len(enhanced_cores) == 0:
        continue

      core_skills = get_core_skills(enhanced_cores)
      if len(core_skills) == 0:
        continue

      # 분석해서 이름만 저장함 [[스킬1, 스킬2, 스킬3], . . .]
      parsed_core_skills = parse_core_skills(core_skills, generated_core_skills)
      if len(parsed_core_skills) > 0:
        core_skill_names.extend(parsed_core_skills)

    del decompose_tabs
    del generated_core_skills
    gc.collect()

    parse_time = time.time()
    print(f"경과 시간[이미지 분석]: {parse_time - model_time:.3f}초")

    end_time = time.time()
    print(f"소요 시간: {end_time - start_time:.3f}초, 분석된 코어의 수: {len(core_skill_names)}")

    if len(core_skill_names) == 0:
      return JSONResponse(content={
        "success": False,
        "message": "이미지에서 쓸만한 코어가 발견되지 않았어요.\n다시 한번 확인해 주세요."
      })


    return JSONResponse(content={
      "success": True,
      "core_skill_names": core_skill_names
    })
  
  except Exception as e:
    import traceback
    traceback.print_exc()

    return JSONResponse(status_code=500, content={
      "success": False
    })

def get_decompose_tabs(displays):
  model_path = "models/get-decompose-tab.pt"
  model = YOLO(model_path)
  results = model.predict(source=displays, verbose=False)

  decompose_tabs = []
  for result in results:
    img = result.orig_img
    boxes = result.boxes.xyxy.cpu().numpy()
    for box in boxes:
      x1, y1, x2, y2 = map(int, box)
      cropped = img[y1:y2, x1:x2]
      decompose_tabs.append(cropped)

  return decompose_tabs

def get_cores(decompose_tab, template):
  cores = []

  gray_decompose_tab = cv2.cvtColor(decompose_tab, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # 윤곽선 검출
  blurred = cv2.GaussianBlur(decompose_tab, (5, 5), 0)
  edges = cv2.Canny(blurred, threshold1=90, threshold2=180)
  contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  for cnt in contours:
    # 소형 객체는 스킵
    area = cv2.contourArea(cnt)
    if area < 500:
      continue

    # 윤곽선 근사화
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 각이 5개가 넘는지 체크
    if 5 <= len(approx):
      x, y, w, h = cv2.boundingRect(approx)
      aspect_ratio = w / h

      # 가로 세로 비율 체크
      if 0.7 < aspect_ratio < 1.3:
        # 코어 부분 추출
        cropped = gray_decompose_tab[y:y+h, x:x+w]
        h, w = cropped.shape

        # 빈 코어와 매칭
        gray_template = cv2.resize(gray_template, (w, h))
        result = cv2.matchTemplate(cropped, gray_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)

        # 빈 코어가 아닐 경우 추출
        if max_val < 0.5:
          cores.append(decompose_tab[y:y+h, x:x+w])

  return cores

def get_enhanced_cores(cores):
  enhanced_cores = []

  dark_color = np.array([0, 0, 0])
  light_color = np.array([94, 115, 113])

  for core in cores:
    # 강화 코어 색으로 마스킹
    masked = cv2.inRange(core, dark_color, light_color)

    # 코어의 배경 좌표 설정
    h, w = masked.shape
    x = w // 6
    y = (h // 2) - 5

    # 코어의 배경 분석
    white = 0
    black = 0
    for i in range(10):
      if masked[y+i, x] == 255:
        white += 1
      else:
        black += 1
    
    # 흰색이 많으면 강화코어
    if white > black:
      enhanced_cores.append(core)

  return enhanced_cores

def get_core_skills(enhanced_cores):
  core_skills = []

  for idx, enhanced_core in enumerate(enhanced_cores):
    hsv = cv2.cvtColor(enhanced_core, cv2.COLOR_BGR2HSV)

    dark_color = np.array([0, 0, 0])
    light_color = np.array([175, 255, 50])

    # 스킬 테두리색으로 마스킹
    masked = cv2.inRange(hsv, dark_color, light_color)

    # 윤곽선 검출
    contours, _ = cv2.findContours(masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선이 없으면 스킵
    if not contours:
      continue

    # 가장 큰 객체 좌표 가져오기
    biggest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(biggest)

    # 스킬 추출
    cropped = enhanced_core[y:y+h, x:x+w]
    core_skills.append(cropped)

  return core_skills

def generate_core_skills(selected_job_class_path):
  def apply_border(core_skill):
    color = [221, 221, 204]
    white = [255, 255, 255]

    for x in range(1, 31):
      core_skill[1, x] = color
      core_skill[30, x] = color
      core_skill[2, x] = white
      core_skill[29, x] = white
    for y in range(2, 30):
      core_skill[y, 1] = color
      core_skill[y, 30] = color
      core_skill[y, 2] = white
      core_skill[y, 29] = white

    return core_skill

  # 직업 스킬들 수집
  job_class_skills = []

  # zipped(스킬 이미지, 스킬 이름)
  for file_name in os.listdir(selected_job_class_path):
    if file_name.lower().endswith(".png"):
      path = os.path.join(selected_job_class_path, file_name)
      image = cv2.imread(path)
      job_class_skills.append((image, file_name[:-4]))

  generated_core_skills = []
  size = 32

  images, file_names = zip(*job_class_skills)
  resized_images = []

  # 스킬 이미지들을 32, 32로 만들기
  for image in images:
    image = cv2.resize(image, (size, size))
    resized_images.append(image)
  
  skills = list(zip(resized_images, file_names))

  # 코어 스킬의 삼각형 마스크 생성
  triangle_mask = np.zeros((size, size), dtype=np.uint8)
  pts = np.array([[0, 0], [31, 0], [16, 16]], np.int32)
  pts = pts.reshape((-1, 1, 2))
  cv2.fillPoly(triangle_mask, [pts], 255)

  # 코어 스킬 생성
  combination_skills = permutations(skills, 3)
  for combo_skill in combination_skills:
    combo_images, combo_file_names = zip(*combo_skill)
    canvas = np.zeros_like(combo_images[0])

    # 중앙(삼각형)에 이미지 배치
    for center in range(3):
      canvas[:, :, center] = np.where(triangle_mask == 255, combo_images[1][:, :, center], canvas[:, :, center])

    # 좌우에 이미지 배치
    for y in range(size):
      for x in range(size):
        # 중앙(삼각형)이 아닐 경우
        if triangle_mask[y, x] == 0:
          if x < 16:
            canvas[y, x] = combo_images[0][y, x]
          else:
            canvas[y, x] = combo_images[2][y, x]
    
    # 강화 코어 테두리 추가
    canvas = apply_border(canvas)
    generated_core_skills.append((canvas, combo_file_names))

    del combo_images
    del canvas
  
  gc.collect()
  return generated_core_skills

def parse_core_skills(core_skills, generated_core_skills):
  def find_matching_image(core_skill, generated_core_skill_images):
    size = 32
    core_skill = cv2.resize(core_skill, (size, size))

    # 모든 조합의 코어 스킬과 비교
    vals = []
    for generated_core_skill_image in generated_core_skill_images:
      generated_core_skill_image = cv2.resize(generated_core_skill_image, (size, size))

      # 만들어진 코어 스킬과 매칭
      result = cv2.matchTemplate(core_skill, generated_core_skill_image, cv2.TM_CCOEFF_NORMED)
      _, max_val, _, _ = cv2.minMaxLoc(result)
      vals.append(max_val)

    # 유사도가 너무 낮을 경우 스킵
    if max(vals) < 0.7:
      return None

    # 가장 유사도가 높은 만들어진 코어 스킬 번호 반환
    return vals.index(max(vals))

  # zipped(코어 스킬 이미지, 코어 스킬 이름)
  generated_core_skill_images, generated_core_skill_names = zip(*generated_core_skills)

  # 넘겨받은 코어 스킬들 분석
  parsed_core_skills = []
  for core_skill in core_skills:
    matched_core_skill_index = find_matching_image(core_skill, generated_core_skill_images)

    if matched_core_skill_index is not None:
      parsed_core_skills.append(generated_core_skill_names[matched_core_skill_index])

  # 메모리 해제
  del generated_core_skill_images
  del generated_core_skill_names

  gc.collect()
  return parsed_core_skills