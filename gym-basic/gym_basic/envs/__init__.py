from gym_basic.envs.basic_env import DroneMaze
from gym_basic.envs.basic_env_2 import BasicEnv2
import matplotlib.pyplot as plt

env = DroneMaze()
obs = env.reset()
plt.imshow(obs)