## 專案介紹

本專案是一個基於 **Flask** 的影像處理應用，提供臉部偵測、單物件與多物件追蹤、臉部馬賽克與五官偵測功能。透過 `OpenCV` 進行影像分析，並透過 API 讓前端或其他應用存取處理後的影像。

## 功能特點

- **臉部偵測 (`face_detection.py`)**：使用 `OpenCV` 偵測影像中的人臉。
- **臉部馬賽克 (`face_mosaic.py`)**：對偵測到的人臉應用馬賽克處理，保護隱私。
- **單物件追蹤 (`single_object_tracking.py`)**：透過 `OpenCV` 的 `TrackerCSRT` 進行 **單個物件追蹤**。
- **多物件追蹤 (`multi_object_tracking.py`)**：使用 `MultiTracker` 同時追蹤 **多個物件**，可追蹤不同區域內的物件。
- **五官偵測 (`detect_features.py`)**：偵測影像中的眼睛、嘴巴和鼻子，並以不同顏色框出。
- **Flask API (`app.py`)**：提供 RESTful API 介面，前端可透過 HTTP 請求進行影像處理。

## 安裝與使用

### 1. 建立虛擬環境並安裝必要的套件

請確保 Python 環境已安裝 `virtualenv`，然後執行以下命令來建立並啟動虛擬環境：

```bash
python -m venv myenv
source myenv/bin/activate  # Windows 使用 myenv\Scripts\activate
pip install -r requirements.txt
```

### 2. 啟動 Flask 伺服器

```bash
python app.py
```

預設伺服器會在 `http://127.0.0.1:5000/` 運行。

### 3. API 端點

#### **臉部偵測**
- **端點**：`POST /detect_face`
- **請求範例**：
  ```json
  {
    "frame": "<base64 encoded image>"
  }
  ```
- **回應範例**：
  ```json
  {
    "faces": [[100, 150, 50, 50], [200, 250, 60, 60]]
  }
  ```

#### **臉部馬賽克**
- **端點**：`POST /apply_mosaic`
- **請求範例**：
  ```json
  {
    "frame": "<base64 encoded image>"
  }
  ```
- **回應範例**：
  ```json
  {
    "frame": "<base64 encoded image with mosaic>"
  }
  ```

#### **單物件追蹤**
- **端點**：`POST /single_object_tracking`
- **請求參數**：
  - `mode`: `initialize`（初始化追蹤）或 `track`（持續追蹤）。
  - `roi`: 目標區域 `[x, y, w, h]`（初始化時必須提供）。
- **請求範例（初始化）**：
  ```json
  {
    "frame": "<base64 encoded image>",
    "mode": "initialize",
    "roi": [100, 150, 50, 50]
  }
  ```
- **請求範例（追蹤）**：
  ```json
  {
    "frame": "<base64 encoded image>",
    "mode": "track"
  }
  ```
- **回應範例**：
  ```json
  {
    "frame": "<base64 encoded tracked image>"
  }
  ```

#### **多物件追蹤**
- **端點**：`POST /multi_object_tracking`
- **請求參數**：
  - `mode`: `initialize`（初始化追蹤）或 `track`（持續追蹤）。
  - `rois`: 目標區域陣列 `[[x1, y1, w1, h1], [x2, y2, w2, h2]]`（初始化時必須提供）。
- **請求範例（初始化）**：
  ```json
  {
    "frame": "<base64 encoded image>",
    "mode": "initialize",
    "rois": [[100, 150, 50, 50], [200, 250, 60, 60]]
  }
  ```
- **請求範例（追蹤）**：
  ```json
  {
    "frame": "<base64 encoded image>",
    "mode": "track"
  }
  ```
- **回應範例**：
  ```json
  {
    "frame": "<base64 encoded tracked image>"
  }
  ```

#### **五官偵測**
- **端點**：`POST /detect_features`
- **功能**：偵測影像中的眼睛（綠色框）、嘴巴（紅色框）和鼻子（藍色框）。
- **請求範例**：
  ```json
  {
    "frame": "<base64 encoded image>"
  }
  ```
- **回應範例**：
  ```json
  {
    "frame": "<base64 encoded image with detected features>"
  }
  ```

## 技術細節

### 1. **Flask API**
- 透過 `Flask` 來架設 HTTP 伺服器，提供 RESTful API。
- 使用 `request.json` 解析前端傳送的 **Base64 影像**。

### 2. **OpenCV 影像處理**
- **臉部偵測**：使用 `haarcascade_frontalface_default.xml` 進行人臉識別。
- **馬賽克處理**：透過 `cv2.resize` 來縮放影像區域並模糊化。
- **單物件追蹤**：使用 `cv2.TrackerCSRT_create()` 來進行 **單物件追蹤**。
- **多物件追蹤**：使用 `cv2.MultiTracker_create()` 來管理 **多個物件的追蹤**。
- **五官偵測**：結合多個 Haar Cascade 模型來偵測眼睛、嘴巴和鼻子，並以不同顏色標註。

## 授權條款

本專案採用 **MIT License**，允許自由使用與修改，惟請保留原始授權條款。

