import sys
import cv2
import mediapipe as mp
import pyttsx3
import math
import threading
import time
import os
from docx import Document
from docx.shared import Inches

from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

# 转换系数（如需用于其他距离计算，此处保留，仅作参考）
PIXELS_PER_CM = 37.0

class PoseApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("姿势检测")
        self.setGeometry(100, 100, 800, 600)  # 设定初始窗口大小

        # 建立主界面布局
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # 视频显示标签（最大化窗口时自动扩展、自动缩放内容）
        self.video_label = QLabel("视频画面")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setScaledContents(True)
        layout.addWidget(self.video_label)

        # “开始检测”按钮
        self.start_button = QPushButton("开始检测")
        self.start_button.clicked.connect(self.start_detection)
        layout.addWidget(self.start_button)

        # 初始化 mediapipe Pose 模型
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        
        # 初始化语音引擎（pyttsx3）
        self.engine = pyttsx3.init()

        # 摄像头与定时器
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 状态控制变量
        self.last_spoken = {"arms": 0, "knee": 0, "ankle": 0, "first_posture": 0}
        self.cooldown = 2  # 提示间隔（秒）
        self.has_spoken_first = False
        self.is_first_message_spoken = False
        self.start_tracking = False

        self.hip_angle_history = []  # 保存最近 3 秒内髋关节角度数据（格式：(时间戳, 角度)）
        self.stability_start_time = None
        self.capture_triggered = False
        self.last_time = time.time()

    def calculate_angle(self, a, b, c):
        """
        计算由点 a, b, c 构成的角度，其中 b 为顶点，a、b、c 均为 [x, y] 坐标
        """
        ab = [a[0] - b[0], a[1] - b[1]]
        bc = [c[0] - b[0], c[1] - b[1]]
        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        magnitude_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
        if magnitude_ab * magnitude_bc == 0:
            return 0
        cos_angle = dot_product / (magnitude_ab * magnitude_bc)
        cos_angle = max(min(cos_angle, 1.0), -1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def speak(self, text):
        """采用多线程播报语音，避免阻塞主循环"""
        threading.Thread(target=lambda: (self.engine.say(text), self.engine.runAndWait())).start()

    def draw_point(self, point, frame_width, frame_height):
        """将归一化坐标转换为像素坐标"""
        return (int(point[0] * frame_width), int(point[1] * frame_height))

    def start_detection(self):
        """点击开始检测后初始化摄像头并启动定时器"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("无法打开摄像头！")
            return
        self.timer.start(30)  # 每 30 毫秒更新一次画面
        self.start_button.setEnabled(False)

    def generate_report(self, captured_image, judgement):
        """生成包含检测结果与图片的 Word 报告，并保存到桌面"""
        report_image_filename = "final_trajectory.jpg"
        cv2.imwrite(report_image_filename, captured_image)
        doc = Document()
        doc.add_heading('姿态检测报告', 0)
        doc.add_paragraph(f'判断结果：{judgement}')
        doc.add_paragraph('拍摄照片如下：')
        doc.add_picture(report_image_filename, width=Inches(4))
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        report_filename = os.path.join(desktop_path, f'Pose_Report_{int(time.time())}.docx')
        doc.save(report_filename)
        self.speak("报告已生成并保存至桌面")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        # 为生成报告保存一份原始帧（未绘制额外信息）
        frame_original = frame.copy()
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        current_time = time.time()

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # 获取左侧关键点
            left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            left_knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            left_ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            left_foot_index = [landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                               landmarks[self.mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]

            # 绘制骨架连线
            cv2.line(frame, self.draw_point(left_shoulder, frame_width, frame_height),
                     self.draw_point(left_hip, frame_width, frame_height), (0, 255, 0), 2)
            cv2.line(frame, self.draw_point(left_hip, frame_width, frame_height),
                     self.draw_point(left_knee, frame_width, frame_height), (0, 255, 0), 2)
            cv2.line(frame, self.draw_point(left_knee, frame_width, frame_height),
                     self.draw_point(left_ankle, frame_width, frame_height), (0, 255, 0), 2)
            cv2.line(frame, self.draw_point(left_ankle, frame_width, frame_height),
                     self.draw_point(left_foot_index, frame_width, frame_height), (0, 255, 0), 2)

            # 计算各关节角度
            knee_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            ankle_angle = self.calculate_angle(left_knee, left_ankle, left_foot_index)
            # 对脚踝角度减10度
            adjusted_ankle_angle = ankle_angle - 10
            hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)

            # 在视频画面上显示角度数值
            cv2.putText(frame, f"Knee: {knee_angle:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ankle: {ankle_angle:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Hip: {hip_angle:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 判断腿部姿势要求（基础条件）
            baseline_leg = (173 <= knee_angle <= 180) and (100 <= adjusted_ankle_angle <= 120)
            if baseline_leg and (not self.has_spoken_first) and (not self.is_first_message_spoken):
                if current_time - self.last_spoken["first_posture"] > self.cooldown:
                    self.speak("请坐直并将双手打平")
                    self.has_spoken_first = True
                    self.is_first_message_spoken = True
                    self.last_spoken["first_posture"] = current_time

            if self.is_first_message_spoken:
                if baseline_leg and (80 <= hip_angle <= 100):
                    if current_time - self.last_spoken["arms"] > self.cooldown:
                        self.speak("请将双手向前伸展")
                        self.last_spoken["arms"] = current_time
                        if not self.start_tracking:
                            self.start_tracking = True
                if not (173 <= knee_angle <= 180):
                    if current_time - self.last_spoken["knee"] > self.cooldown:
                        self.speak("膝关节未伸直")
                        self.last_spoken["knee"] = current_time
                if not (90 <= adjusted_ankle_angle <= 110):
                    if current_time - self.last_spoken["ankle"] > self.cooldown:
                        self.speak("脚踝未勾起")
                        self.last_spoken["ankle"] = current_time

            # 更新髋关节角度历史数据（保留最近 3 秒数据）
            self.hip_angle_history.append((current_time, hip_angle))
            self.hip_angle_history = [(t, a) for (t, a) in self.hip_angle_history if current_time - t <= 3]

            # 当开始记录（start_tracking 为 True）且髋关节角度小于 80°时判断稳定性
            if self.start_tracking and hip_angle < 80:
                if self.hip_angle_history:
                    angles = [a for (_, a) in self.hip_angle_history]
                    if max(angles) - min(angles) < 5:
                        if self.stability_start_time is None:
                            self.stability_start_time = self.hip_angle_history[0][0]
                        stable_duration = current_time - self.stability_start_time
                        cv2.putText(frame, f"Stability: {stable_duration:.1f} s", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        if stable_duration >= 5 and not self.capture_triggered:
                            self.capture_triggered = True
                            captured_image_original = frame_original.copy()
                            # 在 captured_image_original 上重绘骨架与角度信息（用于报告）
                            cv2.line(captured_image_original, self.draw_point(left_shoulder, frame_width, frame_height),
                                     self.draw_point(left_hip, frame_width, frame_height), (0,255,0), 2)
                            cv2.line(captured_image_original, self.draw_point(left_hip, frame_width, frame_height),
                                     self.draw_point(left_knee, frame_width, frame_height), (0,255,0), 2)
                            cv2.line(captured_image_original, self.draw_point(left_knee, frame_width, frame_height),
                                     self.draw_point(left_ankle, frame_width, frame_height), (0,255,0), 2)
                            cv2.line(captured_image_original, self.draw_point(left_ankle, frame_width, frame_height),
                                     self.draw_point(left_foot_index, frame_width, frame_height), (0,255,0), 2)
                            cv2.putText(captured_image_original, f"Knee: {knee_angle:.1f}", (10, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            cv2.putText(captured_image_original, f"Ankle: {ankle_angle:.1f}", (10, 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            cv2.putText(captured_image_original, f"Hip: {hip_angle:.1f}", (10, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                            # 此处可加入更多逻辑以判断最终结果，这里直接使用默认判定
                            judgement = "请重新测试"
                            self.generate_report(captured_image_original, judgement)
                            cv2.putText(frame, "Report Generated!", (50, frame_height - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                            self.speak("报告已生成并保存至桌面")
                else:
                    self.stability_start_time = None
            else:
                self.stability_start_time = None

        # 将最终处理后的帧转换为 QImage 并显示在 video_label 中
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_display.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_display.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoseApp()
    window.show()
    sys.exit(app.exec_())