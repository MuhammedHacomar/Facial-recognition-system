import sys
import os
import cv2
import numpy as np
import face_recognition
import pickle
import pandas as pd
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QLabel, QWidget, QVBoxLayout, QPushButton,
    QHBoxLayout, QDialog, QTableWidget, QTableWidgetItem,
    QCheckBox, QHeaderView, QLineEdit, QMessageBox,
    QProgressBar, QFrame, QGridLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from datetime import datetime, date

# ---------------- YOLO + Encodings ----------------
model = YOLO("runs/detect/train4/weights/best.pt")

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

TOLERANCE = 0.38

# Yüz veri seti yolu
DATASET_PATH = "face_dataset"
os.makedirs(DATASET_PATH, exist_ok=True)

# ---------------- Yoklama CSV ----------------
ATT_FILE = "attendance.csv"

today_str = date.today().strftime("%Y-%m-%d")

# Yoklama dosyasını başlat + günlük sıfırlama
try:
    df = pd.read_csv(ATT_FILE)

    # Temel sütunların var olduğundan emin ol
    if "status" not in df.columns:
        df["status"] = "Yok"
    if "last_seen" not in df.columns:
        df["last_seen"] = ""
    if "last_reset" not in df.columns:
        df["last_reset"] = today_str

    # Eğer gün değiştiyse → yoklamayı sıfırla
    if str(df["last_reset"].iloc[0]) != today_str:
        df["status"] = "Yok"
        df["last_reset"] = today_str

    df.to_csv(ATT_FILE, index=False)

except:
    unique_names = sorted(list(set(known_face_names)))
    df = pd.DataFrame({
        "name": unique_names,
        "status": ["Yok"] * len(unique_names),
        "last_seen": ["" for _ in unique_names],
        "last_reset": [today_str] * len(unique_names)
    })
    df.to_csv(ATT_FILE, index=False)


def sync_attendance_with_known_names():
    """Encodings'teki tüm isimlerin CSV'de olduğundan emin ol."""
    global known_face_names
    df = pd.read_csv(ATT_FILE)

    existing = set(df["name"].tolist())
    added = False

    for n in set(known_face_names):
        if n not in existing:
            df.loc[len(df)] = {
                "name": n,
                "status": "Yok",
                "last_seen": "",
                "last_reset": today_str
            }
            added = True

    if added:
        df.to_csv(ATT_FILE, index=False)


def rebuild_encodings():
    """Face_dataset klasöründen encodings.pkl'i yeniden oluştur ve global değişkenleri güncelle."""
    global known_face_encodings, known_face_names

    new_encodings = []
    new_names = []

    if not os.path.isdir(DATASET_PATH):
        os.makedirs(DATASET_PATH, exist_ok=True)

    for person_name in os.listdir(DATASET_PATH):
        person_folder = os.path.join(DATASET_PATH, person_name)
        if not os.path.isdir(person_folder):
            continue

        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(image)
            if len(encs) == 0:
                continue

            new_encodings.append(encs[0])
            new_names.append(person_name)

    # Global değişkenleri güncelle
    known_face_encodings = new_encodings
    known_face_names = new_names

    # pkl dosyasına kaydet
    data = {"encodings": known_face_encodings, "names": known_face_names}
    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    # Yoklama dosyası ile senkronize et
    sync_attendance_with_known_names()


def mark_attendance(name):
    df = pd.read_csv(ATT_FILE)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    df.loc[df["name"] == name, "status"] = "Var"
    df.loc[df["name"] == name, "last_seen"] = now_str
    df.to_csv(ATT_FILE, index=False)


# ---------------- Başarılı Popup ----------------
class AttendancePopup(QDialog):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yoklama Kaydedildi")
        self.setStyleSheet("background-color:#222; color:white;")

        msg = QLabel(f"✔ Yoklama kaydedildi:\n{name}")
        msg.setStyleSheet("font-size:26px; font-weight:bold; margin:20px; text-align:center;")

        self.newBtn = QPushButton("Yeni Kayıt")
        self.newBtn.setStyleSheet("background-color:#0078ff; padding:12px; font-size:18px;")

        self.exitBtn = QPushButton("Uygulamayı Kapat")
        self.exitBtn.setStyleSheet("background-color:#ff3333; padding:12px; font-size:18px;")

        layout = QVBoxLayout()
        layout.addWidget(msg)
        layout.addWidget(self.newBtn)
        layout.addWidget(self.exitBtn)
        self.setLayout(layout)
        self.setFixedSize(350, 250)


# ---------------- Bilinmeyen Popup ----------------
class UnknownPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Kişi Tanınmadı")
        self.setStyleSheet("background-color:#222; color:white;")

        msg = QLabel("❌ Kişi tanınmadı")
        msg.setStyleSheet("font-size:26px; font-weight:bold; margin:20px; text-align:center;")

        self.newBtn = QPushButton("Yeni Kayıt")
        self.newBtn.setStyleSheet("background-color:#0078ff; padding:12px; font-size:18px;")

        self.exitBtn = QPushButton("Kapat")
        self.exitBtn.setStyleSheet("background-color:#ff3333; padding:12px; font-size:18px;")

        layout = QVBoxLayout()
        layout.addWidget(msg)
        layout.addWidget(self.newBtn)
        layout.addWidget(self.exitBtn)
        self.setLayout(layout)
        self.setFixedSize(350, 250)


# ------------- Thread için Yeni Sınıf: Encoding Yeniden Oluşturma -------------
class RebuildEncodingThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

    def run(self):
        global known_face_encodings, known_face_names
        
        new_encodings = []
        new_names = []
        total_files = 0
        processed_files = 0

        if not os.path.isdir(DATASET_PATH):
            os.makedirs(DATASET_PATH, exist_ok=True)

        # Toplam dosya sayısını hesapla
        for person_name in os.listdir(DATASET_PATH):
            person_folder = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_folder):
                continue
            total_files += len([f for f in os.listdir(person_folder) 
                              if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        if total_files == 0:
            self.progress_signal.emit(100)
            self.finished_signal.emit()
            return

        for person_name in os.listdir(DATASET_PATH):
            person_folder = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_folder):
                continue

            for img_name in os.listdir(person_folder):
                img_path = os.path.join(person_folder, img_name)
                if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue

                image = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(image)
                if len(encs) == 0:
                    processed_files += 1
                    progress = int((processed_files / total_files) * 100)
                    self.progress_signal.emit(progress)
                    continue

                new_encodings.append(encs[0])
                new_names.append(person_name)
                
                processed_files += 1
                progress = int((processed_files / total_files) * 100)
                self.progress_signal.emit(progress)

        # Global değişkenleri güncelle
        known_face_encodings = new_encodings
        known_face_names = new_names

        # pkl dosyasına kaydet
        data = {"encodings": known_face_encodings, "names": known_face_names}
        with open("encodings.pkl", "wb") as f:
            pickle.dump(data, f)

        # Yoklama dosyası ile senkronize et
        sync_attendance_with_known_names()
        
        self.finished_signal.emit()


# ------------- Dialog: Yeni Kullanıcı Adı Girişi -------------
class NewUserNameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Yeni Kullanıcı Ekle")
        self.setStyleSheet("background-color:#222; color:white;")

        label = QLabel("Yeni kullanıcı adını girin:")
        label.setStyleSheet("font-size:20px; margin-bottom:10px;")

        self.name_edit = QLineEdit()
        self.name_edit.setStyleSheet(
            "padding:8px; font-size:18px; border-radius:6px; border:1px solid #555; background-color:#333; color:white;"
        )

        self.okBtn = QPushButton("Devam Et")
        self.okBtn.setStyleSheet("background-color:#0078ff; padding:10px; font-size:18px;")
        self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton("Geri")
        self.cancelBtn.setStyleSheet("background-color:#555; padding:10px; font-size:18px;")
        self.cancelBtn.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.cancelBtn)
        btn_layout.addWidget(self.okBtn)

        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.name_edit)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.setFixedSize(400, 200)

    def on_ok(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Hata", "Lütfen kullanıcı adı girin.")
            return
        self.accept()

    def get_name(self):
        return self.name_edit.text().strip()


# ------------- Dialog: Yeni Kullanıcı Fotoğraf Çekme -------------
class CaptureUserDialog(QDialog):
    def __init__(self, username, parent=None):
        super().__init__(parent)
        self.username = username
        self.capturing = False  # Çekim durumunu takip etmek için

        self.setWindowTitle(f"Fotoğraf Çekme: {username}")
        self.setStyleSheet("background-color:#101010; color:white;")

        # Kamera görüntüsü için label
        self.label = QLabel("Kamera açılıyor...")
        self.label.setFixedSize(640, 480)
        self.label.setStyleSheet("border: 2px solid #444; border-radius: 10px;")

        # Talimatlar label'ı
        self.instructions_label = QLabel(
            "📋 Talimatlar:\n"
            "1. Kameranın önüne geçin\n"
            "2. Yüzünüzün net göründüğünden emin olun\n"
            "3. 'Çekmeye Başlayın' butonuna tıklayın\n"
            "4. Başınızı yavaşça hareket ettirin\n"
            "5. 25 fotoğraf otomatik çekilecek"
        )
        self.instructions_label.setStyleSheet("font-size:14px; padding:10px; background-color:#2a2a2a; border-radius:10px;")
        self.instructions_label.setWordWrap(True)

        # Başlatma butonu
        self.start_button = QPushButton("📸 Çekmeye Başlayın")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #00cc66;
                color: white;
                padding: 15px 30px;
                font-size: 20px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #00b359;
                min-height: 60px;
            }
            QPushButton:hover {
                background-color: #00b359;
                border: 2px solid #00994d;
            }
            QPushButton:pressed {
                background-color: #00994d;
            }
            QPushButton:disabled {
                background-color: #555;
                border: 2px solid #444;
                color: #999;
            }
        """)
        self.start_button.clicked.connect(self.start_capturing)
        self.start_button.setCursor(Qt.PointingHandCursor)

        # İlerleme çubuğu
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 25)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 10px;
                text-align: center;
                font-size: 14px;
                height: 30px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #00cc66;
                border-radius: 8px;
            }
        """)

        self.info = QLabel("Hazır - Çekime başlamak için butona tıklayın")
        self.info.setStyleSheet("font-size:18px; margin-top:10px;")

        # Durum çerçevesi
        status_frame = QFrame()
        status_frame.setStyleSheet("background-color:#2a2a2a; border-radius:10px; padding:10px;")
        status_layout = QVBoxLayout()
        status_layout.addWidget(self.info)
        status_layout.addWidget(self.progress_bar)
        status_frame.setLayout(status_layout)

        # İptal butonu
        self.cancel_button = QPushButton("❌ İptal")
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #ff3333;
                color: white;
                padding: 12px 25px;
                font-size: 18px;
                border-radius: 8px;
                border: 2px solid #cc0000;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #cc0000;
                border: 2px solid #990000;
            }
            QPushButton:pressed {
                background-color: #990000;
            }
        """)
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setCursor(Qt.PointingHandCursor)

        # Grid layout oluştur
        grid_layout = QGridLayout()
        grid_layout.addWidget(self.label, 0, 0, 2, 1)
        grid_layout.addWidget(self.instructions_label, 0, 1)
        grid_layout.addWidget(self.start_button, 1, 1)
        grid_layout.addWidget(status_frame, 2, 0, 1, 2)
        grid_layout.addWidget(self.cancel_button, 3, 0, 1, 2)

        self.setLayout(grid_layout)
        self.setFixedSize(1000, 700)

        # Kamera ayarları
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.saved_count = 0
        self.target_count = 25  # Kaydedilecek fotoğraf sayısı

        # Kullanıcı klasörünü hazırla
        self.user_folder = os.path.join(DATASET_PATH, self.username)
        os.makedirs(self.user_folder, exist_ok=True)

    def start_capturing(self):
        """Çekimi başlat"""
        self.capturing = True
        self.start_button.setEnabled(False)
        self.start_button.setText("📸 Çekim Yapılıyor...")
        self.info.setText("Fotoğraflar çekiliyor... 0/25")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # Yüz tespiti için YOLO çalıştır
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # Yüz tespit edildiyse dikdörtgen çiz (çekim yapılıyor veya yapılmıyor olsun)
        if len(boxes) > 0:
            # En büyük yüz
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist()
            idx = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[idx])

            h, w, _ = frame.shape
            x1 = max(0, min(w - 1, x1))
            x2 = max(0, min(w, x2))
            y1 = max(0, min(h - 1, y1))
            y2 = max(0, min(h, y2))

            # Yüzün etrafına dikdörtgen çiz
            if self.capturing and self.saved_count < self.target_count:
                # Çekim yapılıyorsa yeşil dikdörtgen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                
                face_img = frame[y1:y2, x1:x2]
                if face_img.size > 0:
                    # Fotoğrafı kaydet
                    img_name = f"{self.saved_count:03d}.jpg"
                    save_path = os.path.join(self.user_folder, img_name)
                    cv2.imwrite(save_path, face_img)
                    self.saved_count += 1
                    
                    # İlerleme çubuğunu güncelle
                    self.progress_bar.setValue(self.saved_count)
                    self.info.setText(f"Fotoğraflar çekiliyor... {self.saved_count}/{self.target_count}")
            else:
                # Sadece önizleme yapılıyorsa sarı dikdörtgen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # Görüntüyü arayüzde göster
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img).scaled(self.label.width(), self.label.height()))

        # Hedef fotoğraf sayısına ulaşıldıysa → encoding işlemine geç
        if self.capturing and self.saved_count >= self.target_count:
            self.capturing = False
            self.timer.stop()
            self.cap.release()
            self.start_button.setEnabled(True)
            
            # Encoding oluşturma işlemi için yükleme ekranı
            self.show_encoding_progress()

    def show_encoding_progress(self):
        """Encoding oluşturma işlemi için yükleme ekranı"""
        self.label.clear()
        self.label.setText("Yüz verileri işleniyor...\nLütfen bekleyin")
        self.label.setStyleSheet("font-size:24px; font-weight:bold; qproperty-alignment: AlignCenter;")
        
        # Talimatları ve butonu gizle
        self.instructions_label.hide()
        self.start_button.hide()
        
        # İlerleme çubuğu için ayarlar
        self.info.setText("Yüz verileri işleniyor...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #444;
                border-radius: 10px;
                text-align: center;
                font-size: 16px;
                height: 40px;
                background-color: #2a2a2a;
            }
            QProgressBar::chunk {
                background-color: #0078ff;
                border-radius: 8px;
            }
        """)
        
        # Thread başlat
        self.rebuild_thread = RebuildEncodingThread()
        self.rebuild_thread.progress_signal.connect(self.update_encoding_progress)
        self.rebuild_thread.finished_signal.connect(self.encoding_finished)
        self.rebuild_thread.start()

    def update_encoding_progress(self, value):
        """Encoding ilerlemesini güncelle"""
        self.progress_bar.setValue(value)
        self.info.setText(f"Yüz verileri işleniyor... %{value}")

    def encoding_finished(self):
        """Encoding tamamlandığında çağrılır"""
        QMessageBox.information(self, "Tamamlandı", f"{self.username} için {self.target_count} fotoğraf çekildi ve işlendi.")
        self.accept()

    def closeEvent(self, event):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()
        
        # Eğer thread çalışıyorsa durdur
        if hasattr(self, "rebuild_thread") and self.rebuild_thread.isRunning():
            self.rebuild_thread.quit()
            self.rebuild_thread.wait()
            
        event.accept()


# ---------------- Admin Penceresi ----------------
class AdminWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent

        self.setWindowTitle("Kontrol Paneli")
        self.setStyleSheet("background-color:#1f1f1f; color:white;")
        self.setFixedSize(850, 600)

        # Ana başlık
        title = QLabel("📋 Yoklama Yönetimi")
        title.setStyleSheet("font-size:32px; font-weight:bold; margin:15px; color:#4da6ff;")
        title.setAlignment(Qt.AlignCenter)

        # Yeni kullanıcı ekle butonu
        self.addUserBtn = QPushButton()
        self.addUserBtn.setIcon(self.style().standardIcon(self.style().SP_FileDialogNewFolder))
        self.addUserBtn.setText(" Yeni Kullanıcı Ekle")
        self.addUserBtn.setStyleSheet("""
            QPushButton {
                background-color: #00cc66;
                color: white;
                padding: 15px 25px;
                font-size: 20px;
                font-weight: bold;
                border-radius: 12px;
                border: 2px solid #00b359;
                min-height: 60px;
            }
            QPushButton:hover {
                background-color: #00b359;
                border: 2px solid #00994d;
                transform: scale(1.02);
            }
            QPushButton:pressed {
                background-color: #00994d;
                padding: 16px 26px;
            }
            QPushButton:focus {
                outline: none;
                border: 2px solid #80ffbf;
            }
        """)
        self.addUserBtn.setCursor(Qt.PointingHandCursor)
        self.addUserBtn.clicked.connect(self.add_new_user)

        # Tablo
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["İsim", "Durum", "Son Görülme"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setStyleSheet("""
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #4da6ff;
                font-size: 18px;
                font-weight: bold;
                padding: 12px;
                border: 1px solid #444;
                border-radius: 5px;
            }
        """)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #2a2a2a;
                alternate-background-color: #333;
                color: white;
                font-size: 16px;
                border: 2px solid #444;
                border-radius: 8px;
                gridline-color: #444;
            }
            QTableWidget::item {
                padding: 10px;
                border-bottom: 1px solid #444;
            }
            QTableWidget::item:selected {
                background-color: #0078ff;
                color: white;
            }
            QScrollBar:vertical {
                background: #333;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4da6ff;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3399ff;
            }
        """)
        self.table.setAlternatingRowColors(True)

        self.load_users()

        # Kontrol butonları
        self.saveBtn = QPushButton(" Değişiklikleri Kaydet")
        self.saveBtn.setStyleSheet("""
            QPushButton {
                background-color: #0078ff;
                color: white;
                padding: 12px 24px;
                font-size: 18px;
                border-radius: 8px;
                border: 2px solid #0056b3;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0056b3;
                border: 2px solid #004080;
            }
            QPushButton:pressed {
                background-color: #004080;
            }
        """)
        self.saveBtn.clicked.connect(self.save_changes)
        self.saveBtn.setCursor(Qt.PointingHandCursor)

        self.backBtn = QPushButton(" Geri")
        self.backBtn.setStyleSheet("""
            QPushButton {
                background-color: #666;
                color: white;
                padding: 12px 24px;
                font-size: 18px;
                border-radius: 8px;
                border: 2px solid #555;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #777;
                border: 2px solid #666;
            }
            QPushButton:pressed {
                background-color: #555;
            }
        """)
        self.backBtn.clicked.connect(self.go_back)
        self.backBtn.setCursor(Qt.PointingHandCursor)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.saveBtn)
        control_layout.addWidget(self.backBtn)
        control_layout.setSpacing(15)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(title)
        layout.addWidget(self.addUserBtn)
        layout.addWidget(self.table)
        layout.addLayout(control_layout)
        self.setLayout(layout)

    def go_back(self):
        self.close()
        self.parent.resume_camera()

    def load_users(self):
        df = pd.read_csv(ATT_FILE)
        self.table.setRowCount(len(df))

        for i in range(len(df)):
            name_item = QTableWidgetItem(str(df.iloc[i]["name"]))
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 0, name_item)

            chk = QCheckBox()
            chk.setChecked(str(df.iloc[i]["status"]) == "Var")
            chk.setStyleSheet("""
                QCheckBox {
                    spacing: 10px;
                }
                QCheckBox::indicator {
                    width: 24px;
                    height: 24px;
                    border: 2px solid #555;
                    border-radius: 6px;
                    background-color: #333;
                }
                QCheckBox::indicator:checked {
                    background-color: #00cc66;
                    border: 2px solid #00b359;
                    image: url();
                }
                QCheckBox::indicator:hover {
                    border: 2px solid #777;
                }
            """)

            container = QWidget()
            lay = QHBoxLayout(container)
            lay.addWidget(chk)
            lay.setAlignment(Qt.AlignCenter)
            lay.setContentsMargins(0, 0, 0, 0)
            self.table.setCellWidget(i, 1, container)

            last_seen_val = df.iloc[i].get("last_seen", "")
            if str(last_seen_val) == "nan":
                last_seen_val = ""
            last_seen_item = QTableWidgetItem(str(last_seen_val))
            last_seen_item.setFlags(last_seen_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(i, 2, last_seen_item)

    def save_changes(self):
        df = pd.read_csv(ATT_FILE)

        for i in range(self.table.rowCount()):
            name = self.table.item(i, 0).text()
            container = self.table.cellWidget(i, 1)
            chk = container.findChild(QCheckBox)
            df.loc[df["name"] == name, "status"] = "Var" if chk.isChecked() else "Yok"

        df.to_csv(ATT_FILE, index=False)
        self.saveBtn.setText(" Kaydedildi!")
        QTimer.singleShot(1500, lambda: self.saveBtn.setText("💾 Değişiklikleri Kaydet"))

    def add_new_user(self):
        # 1) Kullanıcı adı girişi
        name_dlg = NewUserNameDialog(self)
        if name_dlg.exec_() != QDialog.Accepted:
            return

        new_name = name_dlg.get_name()

        # 2) Talimat mesajı
        QMessageBox.information(
            self,
            "Fotoğraf Çekme Talimatları",
            "Kamera açılacak ve siz hazır olduğunuzda 'Çekmeye Başlayın' butonuna tıklayın.\n\n"
            "Talimatlar:\n"
            "1. Kameranın önünde konumlanın\n"
            "2. Yüzünüzün net göründüğünden emin olun\n"
            "3. Başınızı yavaşça hareket ettirin\n"
            "4. 25 fotoğraf otomatik çekilecek"
        )

        # 3) Fotoğraf çekme diyaloğunu başlat
        cap_dlg = CaptureUserDialog(new_name, self)
        if cap_dlg.exec_() == QDialog.Accepted:
            # Fotoğraflar çekildikten ve encodings yeniden oluşturulduktan sonra
            self.load_users()


# ---------------- Ana Pencere ----------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Yüz Tanıma ile Yoklama Sistemi")
        self.setStyleSheet("background-color:#101010; color:white;")

        self.label = QLabel("Kamera Hazır")
        self.label.setFixedSize(900, 650)

        self.startBtn = QPushButton("Tanımaya Başla")
        self.startBtn.setStyleSheet("background-color:#28a745; padding:10px; font-size:20px;")
        self.startBtn.clicked.connect(self.enable_recognition)

        self.adminBtn = QPushButton("Admin")
        self.adminBtn.setStyleSheet("background-color:#0078ff; padding:10px; font-size:20px;")
        self.adminBtn.clicked.connect(self.open_admin)

        self.exitBtn = QPushButton("Uygulamayı Kapat")
        self.exitBtn.setStyleSheet("background-color:#ff3333; padding:10px; font-size:20px;")
        self.exitBtn.clicked.connect(self.exit_app)

        hbox = QHBoxLayout()
        hbox.addWidget(self.startBtn)
        hbox.addWidget(self.adminBtn)
        hbox.addWidget(self.exitBtn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(hbox)
        self.setLayout(layout)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.detect_enabled = False
        self.popup_showing = False
        self.frame_counter = 0
        
        # Animasyon değişkenleri
        self.scan_line_pos = 0  # Tarama çizgisi pozisyonu
        self.scan_direction = 1  # 1: aşağı, -1: yukarı
        self.face_box = None  # Yüz bbox koordinatları
        self.recognition_active = False  # Tanıma aktif mi?
        self.recognition_start_time = 0  # Tanıma başlangıç zamanı

    def enable_recognition(self):
        self.detect_enabled = True
        self.frame_counter = 0
        self.recognition_active = True
        self.recognition_start_time = datetime.now()
        self.startBtn.setEnabled(False)
        self.startBtn.setText("Tanıma Yapılıyor...")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return

        # YOLO ile yüz tespiti her zaman yapılsın
        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Yüz bulunduysa bbox'ı kaydet
        if len(boxes) > 0:
            areas = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).tolist()
            idx = int(np.argmax(areas))
            x1, y1, x2, y2 = map(int, boxes[idx])
            self.face_box = (x1, y1, x2, y2)
            
            # Tarama çizgisi animasyonu
            if self.recognition_active and self.detect_enabled:
                # Tarama çizgisi pozisyonunu güncelle
                self.scan_line_pos += 5 * self.scan_direction
                
                # Yön değiştir (yukarı-aşağı)
                if self.scan_line_pos > (y2 - y1):
                    self.scan_direction = -1
                elif self.scan_line_pos < 0:
                    self.scan_direction = 1
        else:
            self.face_box = None

        # Yüz tanıma işlemi
        if self.detect_enabled and not self.popup_showing:
            self.frame_counter += 1
            if self.frame_counter > 60 and self.face_box is not None:
                x1, y1, x2, y2 = self.face_box
                face_img = frame[y1:y2, x1:x2]
                
                if face_img.size > 0 and len(known_face_encodings) > 0:
                    enc = face_recognition.face_encodings(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    if len(enc) > 0:
                        distances = face_recognition.face_distance(known_face_encodings, enc[0])
                        best_i = int(np.argmin(distances))

                        if distances[best_i] < TOLERANCE:
                            name = known_face_names[best_i]
                            mark_attendance(name)
                            self.popup_show("success", name)
                        else:
                            self.popup_show("unknown")
                        
                        # Tanıma sonrası durumu sıfırla
                        self.recognition_active = False
                        self.startBtn.setEnabled(True)
                        self.startBtn.setText("Tanımaya Başla")

        # Görsel efektleri uygula
        frame = self.apply_visual_effects(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(img).scaled(self.label.width(), self.label.height()))

    def apply_visual_effects(self, frame):
        """Görsel efektleri uygula (bbox ve tarama çizgisi)"""
        if self.face_box is not None:
            x1, y1, x2, y2 = self.face_box
            
            # Ana bbox çiz
            if self.recognition_active and self.detect_enabled:
                # Tanıma aktifken turuncu bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)
                
                # Köşe efektleri
                corner_length = 20
                thickness = 3
                # Sol üst köşe
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), (0, 255, 255), thickness)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), (0, 255, 255), thickness)
                # Sağ üst köşe
                cv2.line(frame, (x2, y1), (x2 - corner_length, y1), (0, 255, 255), thickness)
                cv2.line(frame, (x2, y1), (x2, y1 + corner_length), (0, 255, 255), thickness)
                # Sol alt köşe
                cv2.line(frame, (x1, y2), (x1 + corner_length, y2), (0, 255, 255), thickness)
                cv2.line(frame, (x1, y2), (x1, y2 - corner_length), (0, 255, 255), thickness)
                # Sağ alt köşe
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), (0, 255, 255), thickness)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), (0, 255, 255), thickness)
                
                # Tarama çizgisi (hareket eden çizgi)
                if self.scan_line_pos is not None:
                    scan_y = y1 + self.scan_line_pos
                    cv2.line(frame, (x1, scan_y), (x2, scan_y), (0, 255, 255), 2)
                    
                    # Tarama çizgisinin ucuna ok efekti
                    cv2.circle(frame, (x2 - 10, scan_y), 8, (0, 255, 255), 2)
                    cv2.circle(frame, (x1 + 10, scan_y), 8, (0, 255, 255), 2)
                
                # "Taranıyor..." metni
                cv2.putText(frame, "Taraniyor...", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                # Normal görüntüleme için mavi bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                

        
        return frame

    def popup_show(self, mode, name=""):
        self.popup_showing = True
        self.timer.stop()

        if mode == "success":
            popup = AttendancePopup(name, self)
        else:
            popup = UnknownPopup(self)

        popup.newBtn.clicked.connect(lambda: (popup.close(), self.resume()))
        popup.exitBtn.clicked.connect(self.exit_app)
        popup.exec_()

    def resume_camera(self):
        # Admin sayfasından sonra kamera penceresini yeniden aç
        self.show()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.detect_enabled = False
        self.popup_showing = False
        self.frame_counter = 0
        self.recognition_active = False
        self.startBtn.setEnabled(True)
        self.startBtn.setText("Tanımaya Başla")
        self.timer.start(30)

    def resume(self):
        self.detect_enabled = False
        self.popup_showing = False
        self.frame_counter = 0
        self.recognition_active = False
        self.startBtn.setEnabled(True)
        self.startBtn.setText("Tanımaya Başla")
        self.timer.start(30)

    def open_admin(self):
        # Kamerayı ve timer'ı tamamen durdur
        self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()

        # Kamera penceresini geçici olarak gizle
        self.hide()

        # Admin panelini aç
        self.admin = AdminWindow(self)
        self.admin.show()

    def exit_app(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        QApplication.quit()

    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        event.accept()


app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())