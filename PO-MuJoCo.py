import gymnasium as gym
import numpy as np

class POMuJoCoWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Ví dụ HalfCheetah-v4 có obs_dim = 17 (8 qpos + 9 qvel)
        # Ta chỉ lấy phần qpos (vị trí/góc) để tạo POMDP
        self.qpos_dim = 8 
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.qpos_dim,), dtype=np.float32
        )

    def observation(self, observation):
        # Che khuất hoàn toàn qvel, chỉ trả về qpos
        return observation[:self.qpos_dim]

# Khởi tạo môi trường
env = gym.make("HalfCheetah-v4")
po_env = POMuJoCoWrapper(env)