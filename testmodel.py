from stable_baselines3 import SAC
import pandas as pd
import numpy as np
from TradingEnvironmentWithCSV import TradingEnvironmentWithQlearning

market_data_df = pd.read_csv('/Users/srikarpoladi/Downloads/Blockhouse-Work-Trial/data/AAPL_Quotes_Data.csv')
market_data_df['bid_ask_spread'] = market_data_df['ask_price_1'] - market_data_df['bid_price_1']
market_data_df['market_volume'] = market_data_df['bid_size_1'] + market_data_df['ask_size_1']
market_data_df['momentum'] = market_data_df['bid_price_1'].diff().fillna(0)
model = SAC.load("sac_trading_model")

env = TradingEnvironmentWithQlearning(initial_shares=500, time_horizon=390, market_data=market_data_df)

# Test the loaded model in the environment
obs = env.reset()
for _ in range(390): 
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}, Done: {done}")
    if done:
        break

import boto3
import sagemaker
from sagemaker import get_execution_role

bucket_name = 'my-model-bucket'
model_artifact_path = 'sac_trading_model/sac_trading_model.zip'

s3 = boto3.client('s3')
s3.upload_file('sac_trading_model.zip', bucket_name, model_artifact_path)

# Deploy the model on SageMaker
role = get_execution_role()
model = sagemaker.model.Model(
    model_data=f"s3://{bucket_name}/{model_artifact_path}",
    role=role,
    framework_version="2.0",
    sagemaker_session=sagemaker.Session()
)

# Deploying the model to a real-time endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='sac-trading-endpoint'
)