# 1. Python 3.11 slim 베이스 이미지 사용
FROM python:3.11-slim

# 2. 환경 변수 설정 (print가 바로 출력되게 함)
ENV PYTHONUNBUFFERED=1

# 3. 작업 디렉터리 설정
WORKDIR /app

# 4. 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 소스 코드 전체 복사
COPY . .

# 6. 서버 실행
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8080", "--timeout", "600", "--capture-output", "--log-level", "info"]