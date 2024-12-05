import argparse
import d3rlpy
from d3rlpy.logging import CombineAdapterFactory, FileAdapterFactory, WanDBAdapterFactory
from d3rlpy.metrics import EnvironmentEvaluator

from d3rlpy.datasets import get_minari
# 获取数据集和环境
import gymnasium_robotics
import gymnasium as gym

# import minari

# gym.register_envs(gymnasium_robotics)



# 定义命令行参数解析器
def parse_args():
        parser = argparse.ArgumentParser(description="Run d3rlpy algorithm with Minari dataset")
        parser.add_argument(
            "--algorithm",
            type=str,
            default="DecisionTransformer",
            choices=['iql', 'cql', 'td3bc', 'dt'],
            help="Algorithm to use (default: DecisionTransformer)"
        )
        parser.add_argument(
            "--compile_graph",
            action="store_true",
            help="Enable graph compilation for better performance"
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=4096,
            help="Batch size for training (default: 4096)"
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda:0",
            help="Device to run training on (default: cuda:0)"
        )
        parser.add_argument(
            "--n_steps",
            type=int,
            default=1000000,
            help="Number of training steps (default: 1000000)"
        )
        parser.add_argument(
            "--n_steps_per_epoch",
            type=int,
            default=3000,
            help="Number of steps per epoch (default: 3000)"
        )
        parser.add_argument(
            "--logger",
            default=True,
            help="Enable logging with WandB and local file"
        )
        parser.add_argument(
            "--wandb_project",
            type=str,
            default="test_logger",
            help="WandB project name (default: test_logger)"
        )
        parser.add_argument(
            "--train",
            action="store_true",
            help="Training mode"
        )
        parser.add_argument(
            "--ckpt_path",
            type=str,
            default=None,
            help="Checkpoint path for loading"
        )
        parser.add_argument(
            "--save_video",
            action="store_true",
            help="Save video during evaluation"
        )
        return parser.parse_args()

    # 根据配置选择算法
def choose_algorithm(args):
    if args.algorithm == "iql":
        algo_config = d3rlpy.algos.IQLConfig(
            compile_graph=args.compile_graph,
            batch_size=args.batch_size
        )
    elif args.algorithm == "cql":
        algo_config = d3rlpy.algos.CQLConfig(
            compile_graph=args.compile_graph,
            batch_size=args.batch_size
        )
    elif args.algorithm == "dt":
        algo_config = d3rlpy.algos.DecisionTransformerConfig(
        compile_graph=args.compile_graph,
        batch_size=args.batch_size
    )
    elif args.algorithm == "td3bc":
        algo_config = d3rlpy.algos.TD3PlusBCConfig(
        compile_graph=args.compile_graph,
        batch_size=args.batch_size
    )
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

    return algo_config.create(device=args.device)

# 主函数
def main():
    args = parse_args()
   # 定义 Logger
    logger_adapter = None
    if args.logger:
        logger_adapter = CombineAdapterFactory([
            FileAdapterFactory(root_dir="d3rlpy_logs"),
            WanDBAdapterFactory(),
        ])

    # 初始化算法
    if args.train:
        # 如果提供了checkpoint路径，从checkpoint加载
        from d3rlpy.datasets import get_minari
        dataset, env = get_minari('D4RL/kitchen/partial-v2', render_mode="rgb_array")

        if args.ckpt_path:
            model = d3rlpy.load_learnable(args.ckpt_path)
        else:
            model = choose_algorithm(args)

        # 训练离线模型
        if args.algorithm != "dt":
            model.fit(
                dataset,
                n_steps=args.n_steps,
                n_steps_per_epoch=args.n_steps_per_epoch,
                evaluators={
                    "environment": EnvironmentEvaluator(env),
                },
                logger_adapter=logger_adapter
            )
        else:
            model.fit(
                dataset,
                n_steps=args.n_steps,
                n_steps_per_epoch=args.n_steps_per_epoch,
                logger_adapter=logger_adapter
            )

        print("Training complete!")

    # 如果是评估模式
    else:
        if not args.ckpt_path:
            raise ValueError("Must provide checkpoint path for evaluation!")

        model = d3rlpy.load_learnable(args.ckpt_path)

        # 创建环境用于评估
        if args.save_video:
            from gymnasium.wrappers import RecordVideo
            import gymnasium
            import numpy as np
            # env = gymnasium.make('FrankaKitchen-v1', render_mode='rgb_array')
            env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave', 'kettle', 'slide cabinet', 'bottom burner', 'light switch'], render_mode='rgb_array') # 'slide cabinet', 'bottom burner', 'top burner', 'light switch'
#
            env = RecordVideo(env, "eval_videos", episode_trigger=lambda x: True)

        # 执行100个episodes的评估，显示每个episode的结果
        print("\nStarting evaluation...")
        total_reward = 0
        for i in range(100):
            observation, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                # 处理Franka Kitchen环境的观察值
                obs_basic = observation['observation']
                obs_microwave = observation['achieved_goal']['microwave']
                obs_kettle = observation['achieved_goal']['kettle']
                obs_bottomburner = observation['achieved_goal']['bottom burner']
                # obs_lightswitch = observation['achieved_goal']['light switch']
                obs_slidecabinet = observation['achieved_goal']['slide cabinet']

                # 拼接所有观察值
                obs = np.concatenate([
                    obs_basic,
                    obs_microwave,
                    obs_kettle,
                    obs_bottomburner,
                    # obs_lightswitch,
                    obs_slidecabinet
                ])

                # 添加batch维度
                obs = np.expand_dims(obs, axis=0)

                action = model.predict(obs)[0]  # 移除batch维度
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            total_reward += episode_reward
            print(f"Episode {i+1}/100: Reward = {episode_reward:.2f}, Average so far = {total_reward/(i+1):.2f}")

        print(f"\nEvaluation complete!")
        print(f"Average reward over 100 episodes: {total_reward/100:.2f}")

        if args.save_video:
            env.close()

if __name__ == "__main__":
    main()
