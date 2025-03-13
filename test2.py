import cv2
import mediapipe as mp
import math
import pyttsx3

# 初始化MediaPipe和pyttsx3
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
engine = pyttsx3.init()

# 用于计算角度的函数
def calculate_angle(a, b, c):
    # a, b, c 为三个点的坐标
    ab = [b[0] - a[0], b[1] - a[1]]  # 向量 AB
    bc = [c[0] - b[0], c[1] - b[1]]  # 向量 BC
    
    # 计算向量的点积
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    
    # 计算向量的叉积
    cross_product = ab[0] * bc[1] - ab[1] * bc[0]
    
    # 计算夹角
    angle = math.degrees(math.atan2(abs(cross_product), dot_product))
    return angle

# 语音播报函数
def speak(message):
    engine.say(message)
    engine.runAndWait()

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 用来控制是否已抬腿
leg_raised = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # 转换为RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 获取人体关键点
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # 提取关节坐标
        landmarks = results.pose_landmarks.landmark
        
        # 获取左肩、左髋、左膝、左踝、左脚大拇指的坐标
        left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y)
        left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP].y)
        left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y)
        left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y)
        left_big_toe = (landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y)

        # 计算髋部、膝部、踝部的角度
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)  # 髋关节角度
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)  # 膝关节角度
        ankle_angle = calculate_angle(left_knee, left_ankle, left_big_toe)  # 踝关节角度

        # 计算补角
        hip_angle_complement = 180 - hip_angle  # 髋关节补角
        knee_angle_complement = 180 - knee_angle  # 膝关节补角
        ankle_angle_complement = 180 - ankle_angle  # 踝关节补角

        # 在图像上显示角度数字
        # 计算适当的位置来标示角度数值
        cv2.putText(frame, f"{hip_angle_complement:.2f}°", 
                    (int(left_hip[0] * frame.shape[1]) + 10, int(left_hip[1] * frame.shape[0]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{knee_angle_complement:.2f}°", 
                    (int(left_knee[0] * frame.shape[1]) + 10, int(left_knee[1] * frame.shape[0]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{ankle_angle_complement:.2f}°", 
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

        # 检查角度范围并语音播报
        if (100 < ankle_angle_complement < 110) and (173 < knee_angle_complement < 180) and (175 < hip_angle_complement < 180):
            speak("请抬腿")
            leg_raised = True

        # 检测用户是否抬腿并检查膝关节
        if leg_raised:
            # 如果髋关节补角小于150度
            if hip_angle_complement < 150:
                if knee_angle_complement < 170:
                    speak("膝关节未伸直")
                elif hip_angle > 55:
                    if hip_angle < 70:
                        speak("正常")
                    if hip_angle > 70:
                        speak("优秀")

    # 显示结果
    cv2.imshow("Pose Detection", frame)

    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

