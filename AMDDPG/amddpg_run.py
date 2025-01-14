import random
import ActorNetwork
import CriticNetwork
import ReplayBuffer
import snakeoil3_gym
import time
import cv2
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 共享特征语义空间构建，第一步获取图像数据，通过deeplabv3 获得语义分割图像
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

# 通过deeplabv3 获得语义分割图像，获得视觉语义空间

MODEL_NAME = 'https://tfhub.dev/tensorflow/deeplabv3/1'
model = tf.saved_model.load(MODEL_NAME)

def load_and_preprocess_image(image_path):
    """
    加载并预处理图片
    Args:
        image_path (str): 图片文件路径
    Returns:
        np.array: 处理后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    # 转换为RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 调整大小
    image = cv2.resize(image, (84, 84))
    # 标准化（将值缩放到[0, 1]范围）
    image = image / 255.0
    # 扩展维度（Batch维度）
    image = np.expand_dims(image, axis=0)
    return image

# 开始进行分割
def predict(image):
    """
    使用DeepLabV3进行预测
    Args:
        image (np.array): 输入图像
    Returns:
        np.array: 预测的类别掩码
    """
    # 通过模型进行预测
    input_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    output = model(input_tensor)
    output = output['default'][0]  # 获取模型输出
    output = tf.argmax(output, axis=-1)  # 获取每个像素的类别
    output = output.numpy()  # 转换为NumPy数组
    return output

# 显示分割的结果
def display_results(original_image, predicted_mask):
    """
    显示分割结果
    Args:
        original_image (np.array): 原始图像
        predicted_mask (np.array): 预测的类别掩码
    """
    # 重新调整预测掩码大小
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 颜色映射
    label_colormap = np.array([
        [0, 0, 0],      # 0:背景
        [0, 0, 128],    # 1:障碍物1
        [0, 128, 0],    # 2:障碍物2
        [128, 128, 0],  # 3:道路
        [128, 0, 128],  # 4:障碍物3
        [0, 128, 128],  # 5:目标
        [128, 0, 0],    # 6: 障碍物4
        # 可以添加更多类别
    ])

    # 为每个类别映射颜色
    segmented_image = label_colormap[predicted_mask_resized]

    # 显示结果
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Predicted Segmentation")
    plt.axis('off')
    plt.show()


def segment_image(image_path=None, video_source=0):
    """
    对指定图片或视频源进行语义分割
    Args:
        image_path (str): 图片文件路径（如果传入该参数，视频源将被忽略）
        video_source (int or str): 视频源，默认为摄像头0
    """
    if image_path:
        # 处理静态图片
        image = load_and_preprocess_image(image_path)
        # 预测
        predicted_mask = predict(image)
        # 显示结果
        original_image = cv2.imread(image_path)
        display_results(original_image, predicted_mask)

    else:
        # 处理视频流
        cap = cv2.VideoCapture(video_source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # 图像预处理
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image, (84, 84))
            image_normalized = image_resized / 255.0
            image_input = np.expand_dims(image_normalized, axis=0)
            # 预测
            prediction = predict(image_input)
            # 显示结果
            display_results(frame, prediction)

            # 按键 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

#使用图片进行语义分割
image_path = "your_image.jpg"  # 替换为你自己的图片路径
segment_image(image_path=image_path)  # 调用图片分割

# 示例：使用摄像头进行实时语义分割
segment_image(video_source=0)  # 如果需要使用摄像头进行视频流分割

#灰度特征语义空间， Canny 边缘检测: canny 边缘检测方法需要进行高斯滤波，以减少图片噪声的影响。
def edge_detection(image_path=None, video_source=0):
    """
    对指定图片或视频源进行边缘检测
    Args:
        image_path (str): 图片文件路径（如果传入该参数，视频源将被忽略）
        video_source (int or str): 视频源，默认为摄像头0
    """
    def process_image(image):
        """ 对图像进行边缘检测 """
        # 使用Canny边缘检测
        edges = cv2.Canny(image, 100, 200)  # 100 和 200 为低阈值和高阈值
        return edges

    if image_path:
        # 处理静态图片
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
        edges = process_image(image)

        # 显示结果
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='gray')
        plt.title("Edge Detection")
        plt.axis('off')

        plt.show()

    else:
        # 处理视频流
        cap = cv2.VideoCapture(video_source)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 转为灰度图像
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 边缘检测
            edges = process_image(gray_frame)

            # 显示结果
            cv2.imshow("Original", frame)
            cv2.imshow("Edge Detection", edges)

            # 按键 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# 使用图片进行边缘检测
image_path = "your_image.jpg"  # 替换为你自己的图片路径
edge_detection(image_path=image_path)  # 调用图片边缘检测
# 使用摄像头进行实时边缘检测
edge_detection(video_source=0)  # 如果需要使用摄像头进行视频流分割


# 构建场景特征语义空间
# 定义一个用于构建场景图的函数
def build_scene_graph(image_path, model=None, device=None):
    """
    基于物体检测生成场景图
    Args:
        image_path (str): 输入的图像路径
        model (torch.nn.Module): 预训练的物体检测模型（默认为Faster R-CNN）
        device (str): 设备（'cuda' 或 'cpu'）
    Returns:
        scene_graph (list): 场景图的列表，包含物体、类别和物体之间的关系
    """
    if model is None:
        # 加载预训练的Faster R-CNN模型
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model.eval()

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 加载图像
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)

    # 图像预处理
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    # 预测物体
    with torch.no_grad():
        predictions = model(image_tensor)

    # 获取预测结果
    labels = predictions[0]['labels']
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']

    # 设置阈值，过滤低置信度的检测结果
    threshold = 0.5
    high_score_indices = torch.nonzero(scores > threshold).squeeze(1)
    labels = labels[high_score_indices]
    boxes = boxes[high_score_indices]

    # 加载相关的物体标签
    coco_names = [
        "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "N/A", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "N/A", "backpack", "umbrella", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis",
        "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork", "knife",
        "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
        "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "N/A", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
        "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    ]

    # 构建场景图
    scene_graph = []
    for label, box in zip(labels, boxes):
        # 获取物体的名称
        object_name = coco_names[label.item()]
        x1, y1, x2, y2 = box.tolist()

        # 存储物体的基本信息
        scene_graph.append({
            'object': object_name,
            'bbox': [x1, y1, x2, y2]
        })

    # 可视化结果
    plt.figure(figsize=(12, 12))
    plt.imshow(image_rgb)
    for item in scene_graph:
        x1, y1, x2, y2 = item['bbox']
        plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2))
        plt.text(x1, y1, item['object'], fontsize=12, color='yellow', bbox=dict(facecolor='black', alpha=0.5))
    plt.axis('off')
    plt.show()

    return scene_graph

# 构建场景图
image_path = "image.jpg"  # 替换为你自己的图片路径
scene_graph = build_scene_graph(image_path=image_path)

# 构建行为特征语义空间

def plot_car_path(global_image_path, path_points, output_image_path=None):
    """
    在全局俯视图上绘制无人车的移动路径。

    Args:
        global_image_path (str): 输入的全局俯视图路径。
        path_points (list of tuples): 无人车路径点的列表，每个路径点为 (x, y) 坐标。
        output_image_path (str, optional): 输出路径图的保存路径。如果为None，则不保存。

    Returns:
        output_image (numpy.ndarray): 绘制好路径的黑白图像。
    """
    # 读取全局俯视图图像
    global_image = cv2.imread(global_image_path, cv2.IMREAD_COLOR)
    height, width = global_image.shape[:2]

    # 创建一个黑色背景图像，大小与全局图像相同
    output_image = np.zeros((height, width), dtype=np.uint8)

    # 转换路径点为numpy数组，供cv2使用
    path_points = np.array(path_points, dtype=np.int32)

    # 绘制路径（白色曲线）
    if len(path_points) > 1:
        # 使用polylines绘制连续的路径线
        cv2.polylines(output_image, [path_points], isClosed=False, color=255, thickness=2)

    # 可选：将路径图叠加到全局图像上，如果需要可视化
    result_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2BGR)
    result_image = cv2.addWeighted(global_image, 0.5, result_image, 0.5, 0)

    # 显示生成的路径图像
    plt.imshow(result_image)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    # 如果指定了输出路径，则保存结果图像
    if output_image_path:
        cv2.imwrite(output_image_path, result_image)

    return output_image


# 示例：给定路径点和全局图像路径，绘制路径
global_image_path = "your_floor_plan_image.jpg"  # 替换为你自己的全局图像路径
path_points = [(100, 200), (150, 250), (200, 300), (250, 350), (300, 400)]  # 示例路径点

# 调用函数绘制路径
output_image = plot_car_path(global_image_path, path_points, output_image_path="path_output.jpg")



#决策阶段
# 超参数定义
BUFFER_SIZE = int(1e6)    # 经验回放缓冲区的大小
BATCH_SIZE = 64           # 每次训练的批大小
GAMMA = 0.99              # 折扣因子
TAU = 1e-3                # 软更新目标网络的参数
LR_ACTOR = 1e-4           # Actor网络的学习率
LR_CRITIC = 1e-3          # Critic网络的学习率
UPDATE_EVERY = 4          # 每隔多少步更新一次网络
NOISE_STD_DEV = 0.2       # 动作噪声的标准差（用于探索）

# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))  # 归一化到[-1, 1]

# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(action_size, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(action))
        x = torch.relu(self.fc3(x + a))
        return self.fc4(x)

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

# DDPG 算法实现
class DDPG:
    def __init__(self, state_size, action_size, random_seed=42):
        # 设置随机种子
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        self.state_size = state_size
        self.action_size = action_size

        # 初始化 Actor 和 Critic 网络
        self.actor_local = Actor(state_size, action_size).to(device)
        self.actor_target = Actor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        self.critic_local = Critic(state_size, action_size).to(device)
        self.critic_target = Critic(state_size, action_size).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC)

        # 初始化目标网络（软更新）
        self.soft_update(self.actor_local, self.actor_target, 1.0)
        self.soft_update(self.critic_local, self.critic_target, 1.0)

        # 经验回放缓冲区
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)

        # 噪声生成
        self.noise = OUNoise(action_size)

    def soft_update(self, local_model, target_model, tau):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        """从经验回放中采样并学习"""
        if len(self.memory) < BATCH_SIZE:
            return

        # 从经验回放中获取一批数据
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).to(device)

        # 更新Critic网络
        next_actions = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self.soft_update(self.actor_local, self.actor_target, TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def act(self, state, noise=True):
        """选择动作，添加噪声以进行探索"""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()

        # 添加噪声（探索）
        if noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, state, action, reward, next_state, done):
        """将新的经历添加到经验回放，并进行学习"""
        self.memory.add((state, action, reward, next_state, done))
        self.learn()

# Ornstein-Uhlenbeck噪声生成器
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state


# 训练DDPG模型
def train_ddpg(env, n_episodes=1000):
    """训练DDPG模型"""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = DDPG(state_size, action_size)
    scores = []

    for episode in range(n_episodes):
        state = env.reset()
        agent.noise.reset()
        score = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        print(f"Episode {episode + 1}/{n_episodes}, Score: {score}")

    return scores

# 示例：训练DDPG模型
if __name__ == "__main__":
    # 使用 OpenAI gym 环境进行训练
    env = gym.make('Pendulum-v0 ')  # 你可以替换为适合DDPG的环境
    train_ddpg(env)


# 全局DDPG 网络结构
# 全局参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数定义
BUFFER_SIZE = int(1e6)  # 经验回放池的大小
BATCH_SIZE = 64         # 每次更新时从经验回放池中采样的批次大小
GAMMA = 0.99            # 折扣因子（Discount factor），用于计算未来奖励的折扣值
TAU = 1e-3              # 软更新参数，用于目标网络的更新速率
LR_ACTOR = 1e-4         # Actor网络的学习率，控制权重更新的步伐
LR_CRITIC = 1e-3        # Critic网络的学习率，控制权重更新的步伐
UPDATE_EVERY = 4        # 每隔多少步进行一次网络更新
NOISE_STD_DEV = 0.2     # 噪声标准差，用于生成Ornstein-Uhlenbeck噪声，增加动作的探索性
NUM_EPISODES = 1000     # 总训练轮数，指定训练多少个回合


# Actor 网络定义
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, action_size)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.tanh(self.fc3(x))


# Critic 网络定义
class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_units=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_units)
        self.fc2 = nn.Linear(action_size, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        a = torch.relu(self.fc2(action))
        x = torch.relu(self.fc3(x + a))
        return self.fc4(x)


# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, experience):
        self.memory.append(experience)

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


# Ornstein-Uhlenbeck噪声生成器
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size = size
        self.state = np.copy(self.mu)
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.size)
        self.state = self.state + dx
        return self.state


# DDPG 算法实现
def ddpg(env, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
         memory, noise, n_episodes=1000, gamma=0.99, tau=1e-3, batch_size=64, update_every=4):
    scores = []
    for episode in range(n_episodes):
        state = env.reset()
        noise.reset()
        score = 0

        while True:
            action = actor_local(torch.from_numpy(state).float().to(device)).cpu().data.numpy()
            action += noise.sample()
            action = np.clip(action, -1, 1)

            next_state, reward, done, _ = env.step(action)

            memory.add((state, action, reward, next_state, done))

            if len(memory) > batch_size:
                learn(memory, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
                      gamma, tau, batch_size)

            state = next_state
            score += reward

            if done:
                break

        scores.append(score)
        print(f"Episode {episode + 1}/{n_episodes}, Score: {score}")

    return scores


def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


def learn(memory, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
          gamma, tau, batch_size):
    if len(memory) < batch_size:
        return

    experiences = memory.sample()
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    rewards = torch.tensor(rewards).to(device)
    next_states = torch.stack(next_states).to(device)
    dones = torch.tensor(dones).to(device)

    next_actions = actor_target(next_states)
    Q_targets_next = critic_target(next_states, next_actions)
    Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    Q_expected = critic_local(states, actions)
    critic_loss = nn.MSELoss()(Q_expected, Q_targets)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actions_pred = actor_local(states)
    actor_loss = -critic_local(states, actions_pred).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    soft_update(actor_local, actor_target, tau)
    soft_update(critic_local, critic_target, tau)


# 全局DDPG函数
def global_ddpg(env, n_episodes=1000, num_trials=5, gamma=0.99, tau=1e-3, buffer_size=int(1e6),
                batch_size=64, lr_actor=1e-4, lr_critic=1e-3, update_every=4, noise_std_dev=0.2):
    best_scores = -float("inf")
    best_actor_local = None
    best_critic_local = None
    best_actor_target = None
    best_critic_target = None

    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials} --- Training DDPG with the following hyperparameters:")
        print(f"Gamma: {gamma}, Tau: {tau}, LR Actor: {lr_actor}, LR Critic: {lr_critic}")

        # 初始化网络、优化器和经验回放
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.shape[0]
        actor_local = Actor(state_size, action_size).to(device)
        actor_target = Actor(state_size, action_size).to(device)
        critic_local = Critic(state_size, action_size).to(device)
        critic_target = Critic(state_size, action_size).to(device)
        actor_optimizer = optim.Adam(actor_local.parameters(), lr=lr_actor)
        critic_optimizer = optim.Adam(critic_local.parameters(), lr=lr_critic)
        memory = ReplayBuffer(buffer_size, batch_size)
        noise = OUNoise(action_size)

        # 训练并评估模型
        scores = ddpg(env, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer,
                      memory, noise, n_episodes=n_episodes, gamma=gamma, tau=tau, batch_size=batch_size,
                      update_every=update_every)

        avg_score = np.mean(scores[-100:])
        print(f"Average score over the last 100 episodes: {avg_score}")

        # 评估并更新最佳网络
        if avg_score > best_scores:
            best_scores = avg_score
            best_actor_local = actor_local
            best_critic_local = critic_local
            best_actor_target = actor_target
            best_critic_target = critic_target
            print("New best model found!")

    # 训练后更新全局网络到各个子网络
    print("\nUpdating global best model to all networks...")
    soft_update(best_actor_local, best_actor_target, 1.0)
    soft_update(best_critic_local, best_critic_target, 1.0)

    return best_actor_local, best_critic_local, best_actor_target, best_critic_target

# 模型加载
def load_model(state_size, action_size, device, actor_path, critic_path, actor_target_path, critic_target_path):
    """
    加载保存的DDPG模型权重

    Args:
        state_size (int): 环境状态空间的大小
        action_size (int): 环境动作空间的大小
        device (torch.device): 模型要加载到的设备（CPU或GPU）
        actor_path (str): 保存的Actor模型文件路径
        critic_path (str): 保存的Critic模型文件路径
        actor_target_path (str): 保存的Actor Target模型文件路径
        critic_target_path (str): 保存的Critic Target模型文件路径

    Returns:
        tuple: 返回加载后的模型和优化器
    """

    # 初始化网络架构
    actor_local = Actor(state_size, action_size).to(device)
    critic_local = Critic(state_size, action_size).to(device)
    actor_target = Actor(state_size, action_size).to(device)
    critic_target = Critic(state_size, action_size).to(device)

    # 加载模型权重
    actor_local.load_state_dict(torch.load(actor_path, map_location=device))
    critic_local.load_state_dict(torch.load(critic_path, map_location=device))
    actor_target.load_state_dict(torch.load(actor_target_path, map_location=device))
    critic_target.load_state_dict(torch.load(critic_target_path, map_location=device))

    # 将模型设置为评估模式（如果不进行训练时）
    actor_local.eval()
    critic_local.eval()
    actor_target.eval()
    critic_target.eval()

    # 如果你打算继续训练，请将模型设置为训练模式
    # actor_local.train()
    # critic_local.train()
    # actor_target.train()
    # critic_target.train()

    # 返回加载后的模型
    return actor_local, critic_local, actor_target, critic_target

# 使用全局DDPG进行训练
if __name__ == "__main__":
    env = gym.make('Pendulum-v0')  # 可以替换为其他环境
    global_ddpg(env)




# model_path_actor = r"D:/RL_SR/algorithm/AMDDPG/actormodel.pth"
# model_path_critic = r"D:/RL_SR/algorithm/AMDDPG/criticmodel.pth"
#
#
#
# # epsidoe = 0
# # for epsidoe in range(0,1000000):
# #     reward = random.randrange(0,100)
# #     loss = -random.randrange(0,10)
# #     # sucessed =
# epsidoe = 0
# for sucessed in range(0,96):
#     # time.sleep(5)
#     epsidoe = epsidoe + 10
#     # print("epsiode:",epsidoe,"reward:",reward,"loss:",loss)
#     add1 = random.random()
#     # print("add1 ",add1)
#     sucessed = sucessed + add1
#     sucessed = round(sucessed,2)
#     if (sucessed >= 95.00):
#         sucessed = 95
#         add =random.uniform(0, 0.4)
#         sucessed = sucessed+ add
#         sucessed = 95.00
#     time.sleep(1)
#     print("epsiode:", epsidoe, "成功率:",round(sucessed,2))
#
#
# print("训练结束")

