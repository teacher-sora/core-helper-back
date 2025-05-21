## 1. 가상환경 생성
```
python -m venv venv
```

## 2. 가상환경 활성화
### Mac/Linux
```
source venv/bin/activate
```
### Windows (CMD)
```
venv\Scripts\activate
```
### Windows (PowerShell)
```
venv\Scripts\Activate.ps1
```

## 3. 패키지 설치
```
pip install -r requirements.txt
```

## 4. FastAPI 서버 실행
```
uvicorn main:app --reload
```