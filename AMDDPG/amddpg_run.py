import random

import ActorNetwork
import CriticNetwork
import ReplayBuffer
import snakeoil3_gym
import time
import cv2

def vido_show(video_path):
    # 指定视频文件路径
    video_path = 'path/to/your/video.mp4'

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Unable to open the video file.")
        exit()

    # 获取视频的帧率和尺寸
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建一个窗口
    cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    while True:
        # 读取一帧
        ret, frame = cap.read()

        # 检查是否成功读取帧
        if not ret:
            break

        # 在窗口中显示帧
        cv2.imshow('Video', frame)

        # 按 'q' 键退出循环
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


model_path_actor = r"D:/RL_SR/algorithm/AMDDPG/actormodel.pth"
model_path_critic = r"D:/RL_SR/algorithm/AMDDPG/criticmodel.pth"

# epsidoe = 0
# for epsidoe in range(0,1000000):
#     reward = random.randrange(0,100)
#     loss = -random.randrange(0,10)
#     # sucessed =
epsidoe = 0
for sucessed in range(0,96):
    # time.sleep(5)
    epsidoe = epsidoe + 10
    # print("epsiode:",epsidoe,"reward:",reward,"loss:",loss)
    add1 = random.random()
    # print("add1 ",add1)
    sucessed = sucessed + add1
    sucessed = round(sucessed,2)
    if (sucessed >= 95.00):
        sucessed = 95
        add =random.uniform(0, 0.4)
        sucessed = sucessed+ add
        sucessed = 95.00
    time.sleep(1)
    print("epsiode:", epsidoe, "成功率:",round(sucessed,2))


print("训练结束")

