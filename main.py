from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from collections import defaultdict, Counter
from itertools import combinations, permutations

import numpy as np
import cv2
import os
import gc
import math

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
async def core_helper(images: list[UploadFile] = File(...), selected_job_class: str = Form(...), selected_skills: list[str] = Form(...)):
  try:
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

    core_skill_names = []

    for display in displays:
      decompose_tab = get_decompose_tab(display, decompose_tab_template)
      cores = get_cores(decompose_tab, empty_core_template)
      enhanced_cores = get_enhanced_cores(cores)
      core_skills = get_core_skills(enhanced_cores)

      # 분석해서 이름만 저장함 [[스킬1, 스킬2, 스킬3], . . .]
      parsed_core_skills = parse_core_skills(core_skills, selected_job_class_path)

      if len(parsed_core_skills) > 0:
        core_skill_names.extend(parsed_core_skills)

    if len(core_skill_names) == 0:
      return JSONResponse(content={
        "success": False,
        "message": "이미지에서 쓸만한 코어가 발견되지 않았어요.\n다시 한번 확인해 주세요."
      })

    combinations = find_combinations(core_skill_names, selected_skills)

    if len(combinations) == 0:
      return JSONResponse(content={
        "success": False,
        "message": "현재 보유한 코어로는 조합이 어려워요.\n조금 더 코어를 모아주세요."
      })

    return JSONResponse(content={
      "success": True,
      "combinations": combinations
    })
  
  except Exception as e:
    import traceback
    traceback.print_exc()

    return JSONResponse(status_code=500, content={
      "success": False,
      "message": "분석 도중 문제가 발생했어요."
    })

def get_decompose_tab(display, template):
  gray_display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
  gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

  # 분해 탭과 매칭
  result = cv2.matchTemplate(gray_display, gray_template, cv2.TM_CCOEFF_NORMED)
  _, _, _, max_loc = cv2.minMaxLoc(result)

  # 매칭된 영역 좌표 가져오기
  h, w = gray_template.shape
  top_left = max_loc
  bottom_right = (top_left[0] + w, top_left[1] + h)

  # 매칭된 영역 추출
  cropped = display[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
  return cropped

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

    # 5 ~ 9각형인지 체크
    if 5 <= len(approx) <= 9:
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
    color = np.array([0, 0, 0])

    # 스킬 테두리색으로 마스킹
    masked = cv2.inRange(hsv, color, color)

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

def parse_core_skills(core_skills, selected_job_class_path):
  # 코어 스킬을 생성하는 함수
  def generate_core_skills(skills):
    # 테두리를 추가하는 함수
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

    core_skills = []
    size = 32

    images, file_names = zip(*skills)
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
      core_skills.append((canvas, combo_file_names))

      del combo_images
      del canvas
    
    gc.collect()
    return core_skills

  # 매칭되는 이미지를 찾는 함수
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

  # ---------------------- 여기서부터 본문
  # 직업 스킬들 수집
  job_class_skills = []

  # zipped(스킬 이미지, 스킬 이름)
  for file_name in os.listdir(selected_job_class_path):
    if file_name.lower().endswith(".png"):
      path = os.path.join(selected_job_class_path, file_name)
      image = cv2.imread(path)
      job_class_skills.append((image, file_name[:-4]))

  # 모든 조합의 코어 스킬 생성
  # zipped(코어 스킬 이미지, 코어 스킬 이름)
  generated_core_skills = generate_core_skills(job_class_skills)
  generated_core_skill_images, generated_core_skill_names = zip(*generated_core_skills)

  # 넘겨받은 코어 스킬들 분석
  parsed_core_skills = []
  for core_skill in core_skills:
    matched_core_skill_index = find_matching_image(core_skill, generated_core_skill_images)

    if matched_core_skill_index is not None:
      parsed_core_skills.append(generated_core_skill_names[matched_core_skill_index])

  # 메모리 해제
  del generated_core_skills
  del generated_core_skill_images
  del generated_core_skill_names

  gc.collect()
  return parsed_core_skills

def find_combinations(core_skills, selected_skills):
  selected_skills_length = len(selected_skills)

  # 메인 스킬들만 인덱싱
  main_skill_indices = defaultdict(list)

  # { A스킬: [1, 5], B스킬: [2, 3, 4], . . . }
  for idx, core_skill in enumerate(core_skills):
    main_skill_indices[core_skill[0]].append(idx)

  # [A스킬, B스킬, . . .]
  all_main_skills = list(main_skill_indices.keys())

  min_case_count = math.ceil(2 * selected_skills_length / 3)
  max_case_count = min(2 * selected_skills_length, len(all_main_skills)) + 1

  valid_combinations = []
  min_len = float('inf')

  # 선택한 스킬들이 2중첩이 될 수 있는 경우만큼 반복
  for count in range(min_case_count, max_case_count):
    for main_skill_combo in combinations(all_main_skills, count):
      unique_combinations = [[]]

      # 메인 스킬이 중복되지 않도록 코어 스킬 조합을 생성
      for main_skill in main_skill_combo:
        new_combos = []

        for prev in unique_combinations:
          for idx in main_skill_indices[main_skill]:
            # [[1], [5]]
            # [[1, 2], [1, 3], [1, 4], [5, 2], [5, 3], [5, 4]]
            # [[1, 2, . . .], [1, 3, . . .], . . .]
            new_combos.append(prev + [idx])

        unique_combinations = new_combos

      # 조건에 맞는 조합만 추출
      for unique_combo in unique_combinations:
        skill_counter = Counter()

        # 조합 속 코어 스킬들의 스킬 카운팅
        for idx in unique_combo:
          skill_counter.update(core_skills[idx])

        # 선택한 모든 스킬이 2중첩 이상일 경우
        if all(skill_counter[selected_skill] >= 2 for selected_skill in selected_skills):
          if len(unique_combo) < min_len:
            min_len = len(unique_combo)
            valid_combinations = [unique_combo]
          elif len(unique_combo) == min_len:
            valid_combinations.append(unique_combo)
          else:
            # 2중첩이 될 수 있는 조합 중
            # 코어의 개수가 최소로 필요한 경우가 이미 있으므로 종료
            break

  # 유효 조합이 없는 경우 종료
  if len(valid_combinations) == 0:
    return []

  exp_per_core = 50
  level_exp_requirements = [0, 55, 125, 210, 310, 425, 555, 700, 860, 1035, 1225, 1430, 1650, 1885, 2135, 2400, 2680, 2975, 3285, 3610, 3950, 4305, 4675, 5060, 5460]

  # 각 코어들의 레벨 총합
  total_core_levels = []

  # 텍스트 형식의 코어 조합
  valid_name_combinations = []

  for valid_combination in valid_combinations:
    total_core_level = 0
    valid_name_combination = [core_skills[core_skill_index] for core_skill_index in valid_combination]
    main_skills = [skill_names[0] for skill_names in valid_name_combination]

    for main_skill in main_skills:
      count = len(main_skill_indices[main_skill])
      core_exp = count * exp_per_core

      # 해당 코어가 최대 몇 레벨인지 계산
      for idx, level_exp_requirement in enumerate(level_exp_requirements):
        if core_exp <= level_exp_requirement:
          total_core_level += idx
          break

    total_core_levels.append(total_core_level)
    valid_name_combinations.append(valid_name_combination)

  max_level = max(total_core_levels)
  top_level_combination_index = total_core_levels.index(max_level)
  
  # 레벨을 가장 높게 강화할 수 있는 코어 조합
  final_combinations = valid_name_combinations[top_level_combination_index]

  return final_combinations