
import numpy as np
import gym
from gym import spaces

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, prices, window_size=10, init_cash=1000.0):
        super().__init__()
        self.prices = np.array(prices, dtype=float)
        self.window_size = window_size
        self.init_cash = float(init_cash)
        self.max_steps = len(self.prices) - window_size - 1
        self.action_space = spaces.Discrete(3)
        obs_dim = window_size + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.cash = self.init_cash
        self.holdings = 0.0
        self._update_portfolio_value()
        return self._get_obs()

    def _get_obs(self):
        window = self.prices[self.current_step:self.current_step+self.window_size]
        price_norm = window / (window[0]+1e-8) - 1.0
        cash_norm = np.array([self.cash/self.init_cash])
        holdings_norm = np.array([self.holdings])
        return np.concatenate([price_norm, cash_norm, holdings_norm]).astype(np.float32)

    def _update_portfolio_value(self):
        current_price = self.prices[self.current_step+self.window_size]
        self.portfolio_value = self.cash + self.holdings * current_price

    def step(self, action):
        done = False
        prev_value = self.portfolio_value
        price = self.prices[self.current_step+self.window_size]
        if action == 1 and self.cash>0:
            self.holdings += self.cash/price
            self.cash = 0.0
        elif action == 2 and self.holdings>0:
            self.cash += self.holdings * price
            self.holdings = 0.0
        self.current_step +=1
        if self.current_step>self.max_steps:
            done=True
        self._update_portfolio_value()
        reward = self.portfolio_value - prev_value
        info = {'portfolio_value': self.portfolio_value}
        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Cash: {self.cash:.2f}, Holdings: {self.holdings:.4f}, Value: {self.portfolio_value:.2f}")
