import pandas as pd
import numpy as np
from benchmark_costs_script import Benchmark


market_data_df = pd.read_csv('/Users/srikarpoladi/Downloads/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv')
market_data_df['bid_ask_spread'] = market_data_df['ask_price_1'] - market_data_df['bid_price_1']
market_data_df['market_volume'] = market_data_df['bid_size_1'] + market_data_df['ask_size_1']
market_data_df['close'] = (market_data_df['bid_price_1'] + market_data_df['ask_price_1']) / 2
market_data_df['momentum'] = market_data_df['bid_price_1'].diff().fillna(0)

benchmark = Benchmark(data=market_data_df) 
twap_trades = benchmark.get_twap_trades(data=market_data_df, initial_inventory=500)
vwap_trades = benchmark.get_vwap_trades(data=market_data_df, initial_inventory=500)

twap_slippage, twap_market_impact = benchmark.simulate_strategy(twap_trades, market_data_df, preferred_timeframe=390)
vwap_slippage, vwap_market_impact = benchmark.simulate_strategy(vwap_trades, market_data_df, preferred_timeframe=390)

results_df = pd.DataFrame({
    'Strategy': ['TWAP', 'VWAP'],
    'Total Slippage': [np.sum(twap_slippage), np.sum(vwap_slippage)],
    'Total Market Impact': [np.sum(twap_market_impact), np.sum(vwap_market_impact)]
})

print("\nBenchmark Results (Slippage and Market Impact):")
print(results_df.to_string(index=False))