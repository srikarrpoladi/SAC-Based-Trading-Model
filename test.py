import pandas as pd
from TradingEnvironmentWithCSV import TradingEnvironmentWithQlearning

market_data_df = pd.read_csv("/Users/srikarpoladi/Downloads/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv")
market_data_df['bid_ask_spread'] = market_data_df['ask_price_1'] - market_data_df['bid_price_1']
market_data_df['market_volume'] = market_data_df['bid_size_1'] + market_data_df['ask_size_1']
market_data_df['momentum'] = market_data_df['bid_price_1'].diff().fillna(0)

# Step 3: (Optional) Simplify the market data for testing or for a simpler environment
market_data_simplified = market_data_df[['bid_ask_spread', 'market_volume', 'momentum']].copy()

env_with_depth = TradingEnvironmentWithQlearning(
    initial_shares=1000,
    time_horizon=390,  # Full trading day in minutes
    market_data=market_data_df  # Use the full dataframe or market_data_simplified if preferred
)

state = env_with_depth.reset()

trading_horizon = 390  # Total time in minutes
time_step_interval = 15  # Time interval for actions
number_of_steps = trading_horizon // time_step_interval  # 390 steps for minute-wise actions

for _ in range(number_of_steps):  # Adjust the number of steps based on your scenario
    action = 200  # Example action to sell 300 shares
    next_state, reward, done, info = env_with_depth.step(action)
    print(f"Next State: {next_state}, Reward: {reward}, Done: {done}, Info: {info}")

    if done:
        break