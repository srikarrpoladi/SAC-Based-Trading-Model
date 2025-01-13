

import pandas as pd
from stable_baselines3 import SAC
from TradingEnvironmentWithCSV import TradingEnvironmentWithQlearning

market_data_df = pd.read_csv('/Users/srikarpoladi/Downloads/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv')
market_data_df['bid_ask_spread'] = market_data_df['ask_price_1'] - market_data_df['bid_price_1']
market_data_df['market_volume'] = market_data_df['bid_size_1'] + market_data_df['ask_size_1']
market_data_df['momentum'] = market_data_df['bid_price_1'].diff().fillna(0)

env = TradingEnvironmentWithQlearning(initial_shares=500, time_horizon=390, market_data=market_data_df)

model = SAC('MlpPolicy', env, ent_coef='auto', learning_rate=0.00005,  batch_size=1024)

model.learn(total_timesteps=1000)

model.save("sac_trading_model")

# Test the trained model
obs = env.reset()
test_output = []
for _ in range(390): 
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    test_output.append((action, reward, done, info))
    if done:
        break

test_output[:5] 
