import sys
import cv2
import math
import time
import mediapipe as mp
import pyttsx3
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, 
                            QLineEdit, QVBoxLayout, QWidget, 
                            QPushButton, QMessageBox,QHBoxLayout)

# 初始化MediaPipe姿势识别模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 初始化语音引擎
engine = pyttsx3.init()

def calculate_angle(a, b, c):
    """计算三个关节点的角度"""
    ab = [b[0] - a[0], b[1] - a[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    cross_product = ab[0] * bc[1] - ab[1] * bc[0]
    
    angle = math.degrees(math.atan2(abs(cross_product), dot_product))
    return angle

class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    save_signal = pyqtSignal()
    speak_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    capture_signal = pyqtSignal()  

    def __init__(self):
        super().__init__()
        self.subject_id = ""
        self.stable_start = None
        self.target_angle = None
        self.current_frame = None
        self.running = True
        self.leg_raised = False
        self.capture_signal.connect(self.manual_capture)

    @pyqtSlot()
    def manual_capture(self):
        """手动捕获当前帧"""
        if self.current_frame is not None:
            self.save_signal.emit()

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_signal.emit("无法打开摄像头")
            return

        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            annotated_frame = frame.copy()

            # 姿势检测处理
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            self.current_frame = frame.copy()

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # 获取关节坐标（左半身）
                    left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
                    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
                    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
                    left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
                    left_big_toe = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)

                    # 计算关节角度（补角）
                    hip_angle = 180 - calculate_angle(left_shoulder, left_hip, left_knee)
                    knee_angle = 180 - calculate_angle(left_hip, left_knee, left_ankle)
                    ankle_angle = 180 - calculate_angle(left_knee, left_ankle, left_big_toe)

                    # 绘制姿势标记
                    # 在BGR帧上绘制，后续需要转换为RGB
                    cv2.putText(frame, f"{hip_angle:.1f} degree", 
                              (int(left_hip[0] * frame.shape[1]) + 10, 
                               int(left_hip[1] * frame.shape[0]) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # 关节连线（代码保持原有逻辑）
                    cv2.putText(frame, f"{hip_angle:.2f} degree", 
                                (int(left_hip[0] * frame.shape[1]) + 10, int(left_hip[1] * frame.shape[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{knee_angle:.2f} degree", 
                                (int(left_knee[0] * frame.shape[1]) + 10, int(left_knee[1] * frame.shape[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{ankle_angle:.2f} degree", 
                                (int(left_ankle[0] * frame.shape[1]) + 10, int(left_ankle[1] * frame.shape[0]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 计算左肩、左髋、左膝、左踝之间的连线
                    cv2.line(frame, (int(left_shoulder[0] * frame.shape[1]), int(left_shoulder[1] * frame.shape[0])), 
                            (int(left_hip[0] * frame.shape[1]), int(left_hip[1] * frame.shape[0])), (0, 255, 0), 2)  # 连接肩膀和髋
                    cv2.line(frame, (int(left_hip[0] * frame.shape[1]), int(left_hip[1] * frame.shape[0])), 
                            (int(left_knee[0] * frame.shape[1]), int(left_knee[1] * frame.shape[0])), (0, 255, 0), 2)  # 连接髋和膝
                    cv2.line(frame, (int(left_knee[0] * frame.shape[1]), int(left_knee[1] * frame.shape[0])), 
                            (int(left_ankle[0] * frame.shape[1]), int(left_ankle[1] * frame.shape[0])), (0, 255, 0), 2)  # 连接膝和踝
                    cv2.line(frame, (int(left_ankle[0] * frame.shape[1]), int(left_ankle[1] * frame.shape[0])), 
                            (int(left_big_toe[0] * frame.shape[1]), int(left_big_toe[1] * frame.shape[0])), (0, 255, 0), 2)  # 连接踝和大拇指

                    # 绘制圆圈表示关键点
                    cv2.circle(frame, (int(left_shoulder[0] * frame.shape[1]), int(left_shoulder[1] * frame.shape[0])), 5, (0, 0, 255), -1)  # 左肩
                    cv2.circle(frame, (int(left_hip[0] * frame.shape[1]), int(left_hip[1] * frame.shape[0])), 5, (0, 255, 255), -1)  # 左髋
                    cv2.circle(frame, (int(left_knee[0] * frame.shape[1]), int(left_knee[1] * frame.shape[0])), 5, (255, 0, 0), -1)  # 左膝
                    cv2.circle(frame, (int(left_ankle[0] * frame.shape[1]), int(left_ankle[1] * frame.shape[0])), 5, (255, 255, 0), -1)  # 左踝
                    cv2.circle(frame, (int(left_big_toe[0] * frame.shape[1]), int(left_big_toe[1] * frame.shape[0])), 5, (255, 0, 255), -1)  # 左脚大拇指

                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, f"kuan: {hip_angle:.2f} degree", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"xi: {knee_angle:.2f} degree", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f"huai: {ankle_angle:.2f} degree", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # 姿势条件判断
                    if (100 < ankle_angle < 110) and (173 < knee_angle < 180) and (175 < hip_angle < 180):
                        self.speak_signal.emit("请抬腿")
                        self.leg_raised = True




                    # if leg_raised:
                    #     # 如果髋关节补角小于150度
                    #     if hip_angle_complement < 150:
                    #         if knee_angle_complement < 170:
                    #             speak("膝关节未伸直")
                    #         elif hip_angle > 55:
                    #             if hip_angle < 70:
                    #                 speak("正常")
                    #             if hip_angle > 70:
                    #                 speak("优秀")


            except Exception as e:
                print(f"姿势检测错误: {str(e)}")

            self.current_frame = annotated_frame

            # 转换视频帧格式
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.change_pixmap.emit(qt_image)

        cap.release()

    def stop(self):
        self.running = False

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_video_thread()

    def init_ui(self):
        self.setWindowTitle("智能姿势检测系统")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("""
            background: #2C3E50; 
            color: #ECF0F1;
            QLineEdit {
                padding: 12px;
                font-size: 16px;
                border: 2px solid #3498DB;
                border-radius: 5px;
            }
            QPushButton {
                background: #3498DB;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #2980B9;
            }
        """)

        # 创建UI组件
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.id_input = QLineEdit(placeholderText="输入受试者编号")
        self.confirm_btn = QPushButton("确认编号", clicked=self.validate_id)
        self.capture_btn = QPushButton("立即拍照", clicked=self.manual_save)
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background: #27AE60;
                padding: 12px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #219A52;
            }
        """)



        # 布局设置
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.id_input)
        top_layout.addWidget(self.confirm_btn)
        top_layout.addWidget(self.capture_btn)

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.image_label, stretch=1)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def manual_save(self):
        """手动保存按钮点击处理"""
        if not self.id_input.text().strip():
            QMessageBox.warning(self, "警告", "请先输入并确认受试者编号")
            return
        self.video_thread.capture_signal.emit()

    def init_video_thread(self):
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap.connect(self.update_frame)
        self.video_thread.save_signal.connect(self.save_image)
        self.video_thread.speak_signal.connect(self.speak)
        self.video_thread.error_signal.connect(self.show_error)
        self.video_thread.start()

    def validate_id(self):
        if not self.id_input.text().strip():
            QMessageBox.warning(self, "输入错误", "受试者编号不能为空")
            return
        QMessageBox.information(self, "验证成功", "编号已确认，可以开始检测")

    @pyqtSlot(QImage)
    def update_frame(self, image):
        self.image_label.setPixmap(
            QPixmap.fromImage(image).scaled(
                1280, 720, 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

    @pyqtSlot()
    def save_image(self):
        subject_id = self.id_input.text().strip()
        if not subject_id:
            self.speak("未识别到受试者编号")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_type = "manual" if self.sender() == self.capture_btn else "auto"
        filename = f"{subject_id}_{save_type}_{timestamp}.png"
        cv2.imwrite(filename, self.video_thread.current_frame)
        self.speak(f"已保存姿势数据: {filename}")

    @pyqtSlot(str)
    def speak(self, message):
        engine.say(message)
        engine.runAndWait()

    @pyqtSlot(str)
    def show_error(self, message):
        QMessageBox.critical(self, "硬件错误", message)
        self.close()

    def closeEvent(self, event):
        self.video_thread.stop()
        self.video_thread.wait(3000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Arial", 12))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())