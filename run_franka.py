# import d3rlpy
# import wandb
# from gym.wrappers  import RecordVideo
# # from d3rlpy.logger import WandbLogger
# from d3rlpy.datasets import get_minari
# import argparse as arg
# # get config and use it to choose algorithm in d3rlpy
#
# dataset, env = get_minari('D4RL/kitchen/partial-v2')
# # wandb.init(project='test logger ', name='iql')
# # wandb_logger = WandbLogger()
# # prepare algorithm
# logger_adapter = d3rlpy.logging.CombineAdapterFactory([
#    d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
#    d3rlpy.logging.WanDBAdapterFactory(),
# ])
# # iql = d3rlpy.algos.IQLConfig(compile_graph=True, batch_size=4096).create(device="cuda:0")
# model = d3rlpy.algos.DecisionTransformerConfig(compile_graph=True, batch_size=4096).create(device="cuda:0")
#
# # train offline
#
# # train online
# # iql.fit_online(env, n_steps=1000000)
# model.fit(
#     dataset,
#     n_steps=1000000,
#     n_steps_per_epoch=3000,
#     evaluators={
#         'environment': d3rlpy.metrics.EnvironmentEvaluator(env), # evaluate with CartPole-v1 environment
#     },
#     logger_adapter=logger_adapter
# )
#
# # ready to control
# # env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), './video')
#
# # evaluate
#
# # iql = d3rlpy.load_learnable('./model_999000.d3')
# # ans =  d3rlpy.metrics.evaluate_qlearning_with_environment(iql, env, n_trials=100)
# # print(ans)
#
import argparse
import d3rlpy
from d3rlpy.logging import CombineAdapterFactory, FileAdapterFactory, WanDBAdapterFactory
from d3rlpy.metrics import EnvironmentEvaluator

from d3rlpy.datasets import get_minari
# 获取数据集和环境
dataset, env = get_minari('D4RL/kitchen/partial-v2')

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

    # 如果需要评估模型
    # Uncomment to load a model and evaluate
    # model = d3rlpy.load_learnable('./model_999000.d3')
    # results = d3rlpy.metrics.evaluate_qlearning_with_environment(model, env, n_trials=100)
    # print(results)

if __name__ == "__main__":
    main()

