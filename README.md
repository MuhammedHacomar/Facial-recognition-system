# Face Recognition Attendance System

A computer vision-based attendance tracking application implemented in Python. The system utilizes face detection and recognition algorithms to identify registered individuals in real time and automatically log their attendance with precise timestamps.

---

## 📌 Features

- **Real-Time Face Recognition:** Detects and verifies faces against pre-registered identities stored in the database.
- **Automated Attendance Logging:** Records attendance entries instantly into `attendance.csv` with names, dates, and timestamps.
- **Precomputed Encodings Storage:** Uses `encodings.pkl` to store known face embeddings for high-speed runtime matching without re-processing images.
- **Dataset Configuration:** Includes `face.yaml` for detection model and dataset mapping (supports WIDER Face formats).
- **Centralized Application Pipeline:** `app.py` serves as the entry point for camera capture, recognition logic, and system handling.

---

## 🛠 Tech Stack

- **Language:** Python 3.8+
- **Computer Vision & Processing:** OpenCV (`opencv-python`), `face_recognition` / Dlib
- **Object / Face Detection:** YOLO / PyTorch (configured via `face.yaml`)
- **Backend / Execution:** Python / Flask (`app.py`)
- **Storage & Formats:** Pickle (`encodings.pkl`), CSV (`attendance.csv`)

---

## 📁 Project Structure

```text
YüzTanıma/
├── app.py              # Main execution script (detection & attendance pipeline)
├── attendance.csv      # Log file containing attendance records
├── encodings.pkl       # Serialized facial embeddings of registered people
├── face.yaml           # Dataset & detection configuration file
├── faces_db/           # Directory storing source images of individuals
│   └── [person_name]/  # Subfolders named after registered individuals
├── .gitignore          # Git ignore specifications
└── README.md           # Project documentation
```

---

## ⚙️ Prerequisites

- Python 3.8 or higher installed on your operating system.
- An operational webcam or connected video input device.
- C++ build tools (required by `dlib` if building from source on Windows).

---

## 🚀 Installation & Setup

### 1. Clone or Extract the Project

Open your terminal or command prompt in the project root directory:

```bash
cd YüzTanıma
```

### 2. Set Up a Virtual Environment (Recommended)

- **On Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **On Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install the required Python libraries:

```bash
pip install opencv-python face-recognition numpy
```

_(Optional: If your configuration uses Ultralytics YOLO models)_

```bash
pip install ultralytics
```

---

## 💻 Usage

### 1. Registering New Faces

Add images of individuals you wish to recognize inside the `faces_db/` folder:

```text
faces_db/
├── person_one/
│   ├── img1.jpg
│   └── img2.jpg
└── person_two/
    └── img1.jpg
```

### 2. Generate Face Encodings

Ensure `encodings.pkl` is generated/updated with the face images stored in `faces_db/`.

### 3. Run the Application

Launch the system using:

```bash
python app.py
```

Press `q` (or the configured exit key) to stop the camera stream.

---

## 📋 Attendance Output (`attendance.csv`)

When an identity is recognized by the camera, their presence is recorded directly into `attendance.csv`:

| Name       | Date       | Time     |
| :--------- | :--------- | :------- |
| John Doe   | 2026-09-29 | 10:04:15 |
| Jane Smith | 2026-09-29 | 10:05:02 |

---

## 📄 License

This project is distributed under the MIT License. See `LICENSE` for details.
