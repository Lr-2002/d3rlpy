import d3rlpy
import wandb
from gym.wrappers  import RecordVideo
# from d3rlpy.logger import WandbLogger
from d3rlpy.datasets import get_minari
dataset, env = get_minari('D4RL/kitchen/partial-v2')
# wandb.init(project='test logger ', name='iql')
# wandb_logger = WandbLogger()
# prepare algorithm
logger_adapter = d3rlpy.logging.CombineAdapterFactory([
   d3rlpy.logging.FileAdapterFactory(root_dir="d3rlpy_logs"),
   d3rlpy.logging.WanDBAdapterFactory(),
])

# iql = d3rlpy.algos.IQLConfig(compile_graph=True, batch_size=4096).create(device="cuda:0")
#
# # train offline
#
# # train online
# # iql.fit_online(env, n_steps=1000000)
# iql.fit(
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
# env = RecordVideo(gym.make("CartPole-v1", render_mode="rgb_array"), './video')
#
# # evaluate

iql = d3rlpy.load_learnable('./model_999000.d3')
ans =  d3rlpy.metrics.evaluate_qlearning_with_environment(iql, env)
print(ans)
