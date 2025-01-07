import torch
import torch.nn as nn

class CriticNetwork(nn.Module):
    def __init__(self, input_size):
        super(CriticNetwork, self).__init__()
        # 这里定义一个简单的全连接网络作为Critic
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 输出一个状态值

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.fc3(x)
        return value


class SafetyModule:
    def __init__(self, input_size):
        # 初始化两个Critic网络，分别处理不同的风险等级
        self.critic_low_risk = CriticNetwork(input_size)
        self.critic_high_risk = CriticNetwork(input_size)

        # 使用相同的优化器更新两个Critic
        self.optimizer = optim.Adam(list(self.critic_low_risk.parameters()) +
                                    list(self.critic_high_risk.parameters()), lr=0.001)

    def select_critic(self, state, risk_level):
        """
        根据风险等级选择合适的Critic网络
        """
        if risk_level == "high":
            return self.critic_high_risk(state)
        else:
            return self.critic_low_risk(state)

    def evaluate_risk(self, state, external_factors):
        """
        根据外部因素和当前状态评估风险等级
        这里只是一个简单的示例，我们假设通过一些简单的阈值来评估风险等级
        """
        if external_factors['obstacle_proximity'] < 0.5:  # 障碍物接近，判定为高风险
            risk_level = "high"
        else:
            risk_level = "low"
        return risk_level

    def train(self, state, reward, next_state, done, external_factors):
        """
        根据风险等级选择对应的Critic，计算损失并更新模型
        """
        # 评估风险等级
        risk_level = self.evaluate_risk(state, external_factors)

        # 获取当前状态对应的Critic评估值
        critic_value = self.select_critic(state, risk_level)

        # 下一状态的价值评估
        next_state_value = self.select_critic(next_state, risk_level).detach()

        # 计算目标值
        target_value = reward + (1 - done) * 0.99 * next_state_value

        # 计算损失
        critic_loss = (critic_value - target_value).pow(2).mean()

        # 反向传播并优化
        self.optimizer.zero_grad()
        critic_loss.backward()
        self.optimizer.step()

        return critic_loss.item()


# 示例：训练过程
def train_safety_module():
    input_size = 4  # 状态空间维度
    safety_module = SafetyModule(input_size)

    # 假设训练时给定的状态和外部因素
    state = np.random.rand(input_size)  # 当前状态
    next_state = np.random.rand(input_size)  # 下一状态
    reward = np.random.rand()  # 当前奖励
    done = np.random.randint(2)  # 是否结束（1表示结束，0表示继续）
    external_factors = {'obstacle_proximity': np.random.rand()}  # 外部因素（障碍物距离等）

    # 转换为Tensor
    state_tensor = torch.tensor(state, dtype=torch.float32)
    next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

    # 训练一个步骤
    loss = safety_module.train(state_tensor, reward, next_state_tensor, done, external_factors)
    print(f"Critic Loss: {loss}")


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, num_agents):
        super(ActorCritic, self).__init__()

        # 定义卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_inputs, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 7 * 7, 256),
            nn.ReLU()
        )

        # 定义策略网络（Actor）
        self.actors = nn.ModuleList([nn.Sequential(
            nn.Linear(256, num_actions),
            nn.Softmax(dim=-1)
        ) for _ in range(num_agents)])

        # 定义价值网络（Critic）
        self.critic = nn.Linear(256, 1)

    def forward(self, x, agent_idx=None):
        x = x / 255.0  # 将像素值范围缩放到[0,1]
        x = self.conv_layers(x)  # 卷积层处理
        x = x.view(-1, 32 * 7 * 7)  # 展平后送入全连接层
        x = self.fc_layers(x)  # 全连接层处理

        if agent_idx is not None:
            actor_out = self.actors[agent_idx](x)  # 策略网络的输出
        else:
            # 如果agent_idx为None，表示需要计算所有Agent的策略概率
            actor_out = torch.stack([actor_net(x) for actor_net in self.actors], dim=1)

        critic_out = self.critic(x)  # 价值网络的输出
        return actor_out, critic_out


    # 自恢复模块
    def self_recovery_module(image, error_threshold=3000):
        """
        根据四旋翼无人机的环境图像判断飞行方向，避免碰撞。

        参数：
        - image: 当前图像数据，形状为 (84, 84, 3)，表示环境图像。
        - error_threshold: 用于检测障碍物的阈值，黑色像素数量超过此阈值时认为该区域有障碍物。

        返回：
        - direction: 返回四旋翼飞行的控制指令。
          控制指令包括：'move_left', 'move_right', 'move_up', 'move_down', 'move_backward', 'move_forward'
        """
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 反转图像颜色，假设障碍物是黑色区域
        _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # 获取图像的中间五个部分：左、中、右、上、下
        height, width = thresholded.shape
        left_region = thresholded[:, 0:width // 3]  # 左区域
        center_region = thresholded[:, width // 3:2 * width // 3]  # 中心区域
        right_region = thresholded[:, 2 * width // 3:]  # 右区域
        top_region = thresholded[0:height // 3, :]  # 上区域
        bottom_region = thresholded[height // 3:, :]  # 下区域

        # 计算每个区域内黑色像素的数量
        left_obstacle = np.sum(left_region == 0)
        right_obstacle = np.sum(right_region == 0)
        top_obstacle = np.sum(top_region == 0)
        bottom_obstacle = np.sum(bottom_region == 0)
        center_obstacle = np.sum(center_region == 0)

        # 判断每个区域的障碍物是否超过阈值
        if left_obstacle > error_threshold:
            direction = "move_right"  # 左边有障碍物，向右转
        elif right_obstacle > error_threshold:
            direction = "move_left"  # 右边有障碍物，向左转
        elif top_obstacle > error_threshold:
            direction = "move_down"  # 上方有障碍物，向下飞
        elif bottom_obstacle > error_threshold:
            direction = "move_up"  # 下方有障碍物，向上飞
        elif center_obstacle > error_threshold:
            direction = "move_backward"  # 前方有障碍物，向后退
        else:
            direction = "move_forward"  # 无障碍物，向前进

        return direction

    # 示例：使用摄像头获取图像并做出决策
    def run_drone():
        # 初始化摄像头
        camera = cv2.VideoCapture(0)

        while True:
            ret, frame = camera.read()
            if not ret:
                print("无法读取图像，退出！")
                break

            # 调用自恢复模块函数来做出决策
            direction = self_recovery_module(frame)
            print(f"飞行指令: {direction}")

            # 显示图像
            cv2.putText(frame, f"Direction: {direction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Drone View", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 释放资源
        camera.release()
        cv2.destroyAllWindows()

