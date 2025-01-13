import gym
from gym import spaces
import pandas as pd
import numpy as np

class TradingEnvironmentWithQlearning(gym.Env):
    def __init__(self, initial_shares, time_horizon, market_data):
        super(TradingEnvironmentWithQlearning, self).__init__()
        self.initial_shares = initial_shares
        self.time_horizon = time_horizon
        self.market_data = market_data
        self.current_step = 0
        self.q_table = {}  
        self.learning_rate = 0.1
        self.discount_factor = 0.9  
        self.action_space = spaces.Box(low=0, high=300, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

    def reset(self):
        self.time_remaining = self.time_horizon
        self.shares_left = self.initial_shares
        self.current_step = 0
        self.state = self._get_state()
        return self.state

    def _get_state(self):
        # State includes time, inventory, and market conditions
        private_state = [self.time_remaining, self.shares_left]
        market_state = [
            float(self.market_data.iloc[self.current_step]["bid_ask_spread"]),
            float(self.market_data.iloc[self.current_step]["market_volume"]),
            float(self.market_data.iloc[self.current_step]["momentum"])
        ]
        return tuple(private_state + market_state)

    def calculate_reward(self, sell_quantity, execution_price, reference_price, market_volume, initial_mid_spread):
        # Calculate slippage cost relative to the ideal mid-spread price
        slippage_cost = abs(execution_price - initial_mid_spread) * (sell_quantity / self.initial_shares)

        if market_volume > 0:
            market_impact_cost = (sell_quantity / market_volume) ** 2 * 100
        else:
            market_impact_cost = 100  

        opportunity_cost = (self.shares_left / self.initial_shares) * 100 if self.time_remaining == 0 else 0

        slippage_tolerance = 0.01  
        if abs(execution_price - reference_price) < slippage_tolerance:
            positive_reward = 10  
        else:
            positive_reward = 0

        total_cost = -(slippage_cost + market_impact_cost + opportunity_cost) + positive_reward

        return total_cost

    def step(self, action):
        sell_quantity = action[0] if isinstance(action, np.ndarray) else action
        self.shares_left -= sell_quantity
        self.time_remaining -= 1
        self.current_step += 1

        if self.shares_left < 0:
            self.shares_left = 0
        
        if self.current_step >= len(self.market_data):
            self.current_step = len(self.market_data) - 1

        next_state = self._get_state()

        execution_price = self.market_data.iloc[self.current_step]["bid_price_1"]
        reference_price = (self.market_data.iloc[self.current_step]["bid_price_1"] + self.market_data.iloc[self.current_step]["ask_price_1"]) / 2
        market_volume = self.market_data.iloc[self.current_step]["market_volume"]
        initial_mid_spread = (self.market_data.iloc[0]["bid_price_1"] + self.market_data.iloc[0]["ask_price_1"]) / 2

        # Calculate the reward based on the difference from ideal execution price
        reward = self.calculate_reward(sell_quantity, execution_price, reference_price, market_volume, initial_mid_spread)

        done = self.time_remaining == 0 or self.shares_left == 0
        info = {"shares_sold": sell_quantity}

        return next_state, reward, done, info

    def update_q_value(self, state, action, reward, next_state):
        # Get the current Q-value for the state-action pair
        current_q = self.q_table.get((state, action), 0.0)
        next_max_q = max([self.q_table.get((next_state, a), 0.0) for a in range(self.action_space.n)])
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount_factor * next_max_q)
        self.q_table[(state, action)] = new_q

    def optimize_actions(self):
        for t in reversed(range(self.time_horizon)):
            for i in range(self.initial_shares + 1):
                state = (t, i)  
                for action in range(self.action_space.n): 
                    next_state, reward, done, info = self.step(action)
                    self.update_q_value(state, action, reward, next_state)
