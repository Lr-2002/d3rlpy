import d3rlpy
import numpy as np
from d3rlpy.datasets import get_minari
from d3rlpy.metrics import evaluate_qlearning_with_environment

# 加载数据集和环境（保持与训练一致）
dataset, env = get_minari('D4RL/kitchen/partial-v2')

# 加载训练好的模型
iql = d3rlpy.load_learnable("./model_999000.d3")

# 评估模型效果：运行 200 步并计算 return
def evaluate_model(env, model, n_episodes=1, max_steps=200):
    total_returns = []
    for _ in range(n_episodes):
        observation, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        while not done and steps < max_steps:
            # 获取模型动作
            action = model.predict(np.array([observation]))[0]
            # 执行动作
            observation, reward, done, _, _  = env.step(action)
            total_reward += reward
            steps += 1
        total_returns.append(total_reward)
    return total_returns

# 运行评估并打印结果
returns = evaluate_model(env, iql, n_episodes=10, max_steps=200)
print(f"Average Return over 10 episodes: {sum(returns) / len(returns):.2f}")
# print('total_return is ', total_returns)

