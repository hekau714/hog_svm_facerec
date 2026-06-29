# HOG + SVM Face Detector (WebSocket)

Realtime face detection app:
- Frontend: camera in browser (`web/index.html`)
- Backend: FastAPI + WebSocket (`server.py`)
- Model: HOG + Linear SVM (`detect.py`, `output/models/svm_face_detector.joblib`)
- Face recognition: HOG + SVM (`face_recognizer.py`, `output/models/face_recognizer.joblib`)

## Local Run

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Open:
- `http://127.0.0.1:8000`

## Face Recognition Training (HOG+SVM)

### 1) Prepare dataset structure

Put identity images in person-per-folder layout:

```text
known_faces/
  Alice/
    img1.jpg
    img2.jpg
  Bob/
    img1.jpg
    img2.jpg
```

### 2) Build cropped face dataset + train recognizer

This uses the face detector to crop face-only images into `known_faces_cropped/`,
then trains the recognition model on those crops.

```bash
python face_recognizer.py --prepare-crops --train --backend hog_svm \
  --source-dataset known_faces \
  --cropped-dataset known_faces_cropped \
  --detector-model output/models/svm_face_detector.joblib
```

### 3) Retrain only (if crops already exist)

```bash
python face_recognizer.py --train --backend hog_svm --dataset known_faces_cropped
```

### 4) Use in realtime server

`server.py` automatically loads:

- `output/models/face_recognizer.joblib`

On startup, check logs for:

- `[recog] Loaded face recognizer: output/models/face_recognizer.joblib`

## Push To GitHub

```bash
git init
git add .
git commit -m "Initial commit: websocket face detector"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

## Deploy On Linux Server

### 1) Clone and install

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd hog_svm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run app

```bash
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
```

### 3) Production (recommended)

- Put Nginx in front of Uvicorn.
- Configure HTTPS in Nginx (Let's Encrypt).
- Proxy `/` and `/ws` to `http://127.0.0.1:8000`.

Minimal WebSocket Nginx location:

```nginx
location /ws {
    proxy_pass http://127.0.0.1:8000/ws;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
}
```

## Notes

- Keep `output/models/svm_face_detector.joblib` in repo or provide your own model path.
- Keep `output/models/face_recognizer.joblib` for realtime name classification.
- Camera access from browser requires HTTPS in many mobile browsers.
